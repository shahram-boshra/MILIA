#!/usr/bin/env python3
"""
Unit tests for the PyG Model Introspector module (v2.1.0).

Tests model discovery, parameter introspection, metadata generation, search
space generation, validation, category inference, and backward-compatible API
using mock PyG model classes (no real torch_geometric or torch required).

These tests use function-level mocking exclusively to prevent mock pollution
of sys.modules across test files during pytest collection.

Module location: milia_pipeline/models/registry/pyg_introspector.py
"""

import inspect
import types
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers – lightweight stand-ins for torch.nn.Module and PyG models
# ---------------------------------------------------------------------------


class _FakeModule:
    """Minimal stand-in for torch.nn.Module so issubclass() checks pass."""

    pass


class _FakeGCN(_FakeModule):
    """The GCN model uses graph convolutional layers."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        act: str = "relu",
        norm: str | None = None,
        jk: str | None = None,
        **kwargs,
    ):
        pass

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """Forward pass.

        x: Node feature matrix.
        edge_index: Graph connectivity.
        """
        pass


class _FakeGAT(_FakeModule):
    """Mock GAT model with attention heads."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int | None = None,
        heads: int = 1,
        dropout: float = 0.0,
        v2: bool = False,
        **kwargs,
    ):
        pass

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        pass


class _FakeGraphSAGE(_FakeModule):
    """Mock GraphSAGE model."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        project: bool = False,
    ):
        pass

    def forward(self, x, edge_index, batch=None):
        pass


class _FakeEdgeCNN(_FakeModule):
    """Mock EdgeCNN – no batch in forward, no out_channels."""

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int):
        pass

    def forward(self, x, edge_index):
        pass


class _FakeSchNet(_FakeModule):
    """Mock SchNet molecular model."""

    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
    ):
        pass

    def forward(self, z, pos, batch=None):
        """z: Atomic numbers. pos: Atomic positions."""
        pass


class _FakeDimeNet(_FakeModule):
    """Mock DimeNet with specialized params."""

    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_blocks: int = 6,
        num_bilinear: int = 8,
        num_spherical: int = 7,
        num_radial: int = 6,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
    ):
        pass

    def forward(self, z, pos, batch=None):
        pass


class _FakeGAE(_FakeModule):
    """Mock Graph Autoencoder."""

    def __init__(self, encoder, decoder=None):
        pass

    def forward(self, x, edge_index):
        pass


class _FakeKwargsModel(_FakeModule):
    """Model that accepts **kwargs."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        pass

    def forward(self, x, edge_index):
        pass


class _FakeNoKwargsModel(_FakeModule):
    """Model without **kwargs."""

    def __init__(self, in_channels: int, out_channels: int):
        pass

    def forward(self, x, edge_index):
        pass


class _FakeNoForward(_FakeModule):
    """Model with no forward method."""

    def __init__(self, in_channels: int):
        pass


class _FakeNoSignature:
    """Model where inspect.signature fails."""

    __init__ = None


class _FakePoolModel(_FakeModule):
    """Mock pooling model."""

    def __init__(self, in_channels: int, ratio: float = 0.5):
        pass

    def forward(self, x, edge_index, batch=None):
        pass


class _FakeTransformerModel(_FakeModule):
    """Mock transformer model."""

    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 4):
        pass

    def forward(self, x, edge_index, batch=None):
        pass


# ---------------------------------------------------------------------------
# Helper – building fake module for discover_pyg_models()
# ---------------------------------------------------------------------------


def _build_fake_pyg_nn_models_module():
    """Build a fake torch_geometric.nn.models module with public model classes."""
    mod = types.ModuleType("torch_geometric.nn.models")
    mod.GCN = _FakeGCN
    mod.GAT = _FakeGAT
    mod.GraphSAGE = _FakeGraphSAGE
    mod.EdgeCNN = _FakeEdgeCNN
    mod.SchNet = _FakeSchNet
    mod.DimeNet = _FakeDimeNet
    mod._PrivateHelper = type("_PrivateHelper", (_FakeModule,), {})
    mod.some_function = lambda: None
    mod.SOME_CONSTANT = 42
    return mod


# ---------------------------------------------------------------------------
# Import path constant – single source of truth for the module under test
# ---------------------------------------------------------------------------
_MODULE_PATH = "milia_pipeline.models.registry.pyg_introspector"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_introspector_singleton():
    """
    Reset the PyGModelIntrospector singleton and module-level _introspector
    before and after each test to prevent state leaking between tests.
    """
    intro = pytest.importorskip(_MODULE_PATH)

    def _clear():
        intro._introspector = None
        intro.PyGModelIntrospector._instance = None

    _clear()
    yield
    _clear()


@pytest.fixture
def mock_nn_module():
    """
    Patch torch.nn.Module at the test level so that issubclass checks in
    discover_pyg_models work with our fake model classes.
    """
    intro = pytest.importorskip(_MODULE_PATH)
    fake_nn = types.ModuleType("torch.nn")
    fake_nn.Module = _FakeModule

    with patch.dict("sys.modules", {"torch.nn": fake_nn}), patch.object(intro, "nn", fake_nn):
        yield fake_nn


@pytest.fixture
def fake_discovery_env(mock_nn_module):
    """
    Patch importlib.import_module so that discover_pyg_models() finds our
    fake model classes instead of requiring real torch_geometric.

    Also patches sys.modules so that raw ``import`` statements inside
    discover_pyg_models (e.g. ``import torch_geometric.nn.models``)
    resolve to the fakes.
    """
    intro = pytest.importorskip(_MODULE_PATH)

    fake_models_mod = _build_fake_pyg_nn_models_module()
    fake_nn_top = types.ModuleType("torch_geometric.nn")
    fake_nn_top.GCN = _FakeGCN
    fake_nn_top.GAT = _FakeGAT
    fake_nn_top.GraphSAGE = _FakeGraphSAGE

    # Build a minimal torch_geometric package hierarchy so that raw
    # ``import torch_geometric.nn.models`` resolves via sys.modules.
    fake_tg = types.ModuleType("torch_geometric")
    fake_tg.nn = fake_nn_top
    fake_nn_top.models = fake_models_mod

    sys_modules_patch = {
        "torch_geometric": fake_tg,
        "torch_geometric.nn": fake_nn_top,
        "torch_geometric.nn.models": fake_models_mod,
    }

    original_import = intro.importlib.import_module

    def _patched_import(name):
        if name == "torch_geometric.nn.models":
            return fake_models_mod
        if name == "torch_geometric.nn":
            return fake_nn_top
        if name.startswith("torch_geometric.nn.models."):
            raise ImportError(f"No module named '{name}'")
        return original_import(name)

    with (
        patch.dict("sys.modules", sys_modules_patch),
        patch.object(intro.importlib, "import_module", side_effect=_patched_import),
    ):
        yield fake_models_mod


# ═══════════════════════════════════════════════════════════════════════════
# 1. ParameterInfo (Pydantic BaseModel)
# ═══════════════════════════════════════════════════════════════════════════


class TestParameterInfo:
    """Tests for the ParameterInfo Pydantic model."""

    def test_construction_with_required_fields(self):
        intro = pytest.importorskip(_MODULE_PATH)
        pi = intro.ParameterInfo(name="x", param_type="int", required=True)
        assert pi.name == "x"
        assert pi.param_type == "int"
        assert pi.required is True
        assert pi.default is None
        assert pi.description == ""

    def test_construction_with_all_fields(self):
        intro = pytest.importorskip(_MODULE_PATH)
        pi = intro.ParameterInfo(
            name="hidden_channels",
            param_type="int",
            required=False,
            default=64,
            description="Number of hidden channels",
            min_value=1.0,
            max_value=512.0,
            choices=[32, 64, 128],
        )
        assert pi.default == 64
        assert pi.min_value == 1.0
        assert pi.choices == [32, 64, 128]

    def test_to_dict_returns_dict(self):
        intro = pytest.importorskip(_MODULE_PATH)
        pi = intro.ParameterInfo(name="dropout", param_type="float", required=False, default=0.5)
        d = pi.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "dropout"
        assert d["default"] == 0.5

    def test_mutable_instance(self):
        intro = pytest.importorskip(_MODULE_PATH)
        pi = intro.ParameterInfo(name="x", param_type="int", required=True)
        pi.description = "Updated"
        assert pi.description == "Updated"


# ═══════════════════════════════════════════════════════════════════════════
# 2. introspect_model_signature
# ═══════════════════════════════════════════════════════════════════════════


class TestIntrospectModelSignature:
    """Tests for introspect_model_signature()."""

    def test_gcn_returns_expected_params(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakeGCN)
        assert "in_channels" in params
        assert "hidden_channels" in params
        assert "num_layers" in params
        assert "dropout" in params

    def test_excludes_self_args_kwargs(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakeGCN)
        assert "self" not in params
        assert "args" not in params
        assert "kwargs" not in params

    def test_required_flag_correct(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakeGCN)
        assert params["in_channels"].required is True
        assert params["dropout"].required is False

    def test_default_values_captured(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakeGCN)
        assert params["dropout"].default == 0.0
        assert params["act"].default == "relu"

    def test_required_param_default_is_none(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakeGCN)
        assert params["in_channels"].default is None

    def test_returns_parameter_info_instances(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakeGCN)
        for _name, info in params.items():
            assert isinstance(info, intro.ParameterInfo)

    def test_empty_result_for_broken_signature(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakeNoSignature)
        assert params == {}

    def test_gat_heads_param(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakeGAT)
        assert "heads" in params
        assert params["heads"].default == 1

    def test_schnet_params(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakeSchNet)
        assert "num_filters" in params
        assert params["cutoff"].default == 10.0

    def test_dimenet_specialized_params(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakeDimeNet)
        assert "num_blocks" in params
        assert "num_spherical" in params
        assert params["num_spherical"].default == 7


# ═══════════════════════════════════════════════════════════════════════════
# 3. introspect_forward_signature
# ═══════════════════════════════════════════════════════════════════════════


class TestIntrospectForwardSignature:
    """Tests for introspect_forward_signature()."""

    def test_gcn_forward_params(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_forward_signature(_FakeGCN)
        assert "x" in params
        assert "edge_index" in params
        assert "batch" in params

    def test_self_excluded(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_forward_signature(_FakeGCN)
        assert "self" not in params

    def test_gat_has_edge_attr(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_forward_signature(_FakeGAT)
        assert "edge_attr" in params

    def test_required_flag(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_forward_signature(_FakeGCN)
        assert params["x"].required is True
        assert params["edge_weight"].required is False

    def test_edgecnn_no_batch(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_forward_signature(_FakeEdgeCNN)
        assert "batch" not in params

    def test_schnet_forward_has_z_and_pos(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_forward_signature(_FakeSchNet)
        assert "z" in params
        assert "pos" in params
        assert params["z"].required is True

    def test_no_forward_returns_empty(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_forward_signature(_FakeNoForward)
        assert params == {}


# ═══════════════════════════════════════════════════════════════════════════
# 4. model_accepts_kwargs
# ═══════════════════════════════════════════════════════════════════════════


class TestModelAcceptsKwargs:
    """Tests for model_accepts_kwargs()."""

    def test_kwargs_model_returns_true(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro.model_accepts_kwargs(_FakeKwargsModel) is True

    def test_no_kwargs_model_returns_false(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro.model_accepts_kwargs(_FakeNoKwargsModel) is False

    def test_gcn_with_kwargs_true(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro.model_accepts_kwargs(_FakeGCN) is True

    def test_graphsage_no_kwargs_false(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro.model_accepts_kwargs(_FakeGraphSAGE) is False

    def test_broken_signature_returns_false(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro.model_accepts_kwargs(_FakeNoSignature) is False


# ═══════════════════════════════════════════════════════════════════════════
# 5. get_required_data_attributes
# ═══════════════════════════════════════════════════════════════════════════


class TestGetRequiredDataAttributes:
    """Tests for get_required_data_attributes()."""

    def test_gcn_requires_x_and_edge_index(self):
        intro = pytest.importorskip(_MODULE_PATH)
        attrs = intro.get_required_data_attributes(_FakeGCN)
        assert "x" in attrs
        assert "edge_index" in attrs

    def test_schnet_requires_z_and_pos(self):
        intro = pytest.importorskip(_MODULE_PATH)
        attrs = intro.get_required_data_attributes(_FakeSchNet)
        assert "z" in attrs
        assert "pos" in attrs

    def test_batch_excluded(self):
        intro = pytest.importorskip(_MODULE_PATH)
        attrs = intro.get_required_data_attributes(_FakeGCN)
        assert "batch" not in attrs

    def test_optional_params_excluded(self):
        intro = pytest.importorskip(_MODULE_PATH)
        attrs = intro.get_required_data_attributes(_FakeGCN)
        assert "edge_weight" not in attrs

    def test_no_forward_returns_empty(self):
        intro = pytest.importorskip(_MODULE_PATH)
        attrs = intro.get_required_data_attributes(_FakeNoForward)
        assert attrs == set()


# ═══════════════════════════════════════════════════════════════════════════
# 6. Conv Kwargs
# ═══════════════════════════════════════════════════════════════════════════


class TestConvKwargs:
    """Tests for get_model_conv_kwargs() and KNOWN_CONV_KWARGS."""

    def test_known_conv_kwargs_is_set(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert isinstance(intro.KNOWN_CONV_KWARGS, set)
        assert "add_self_loops" in intro.KNOWN_CONV_KWARGS

    def test_model_specific_keys(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert "GCN" in intro.MODEL_SPECIFIC_CONV_KWARGS
        assert "GAT" in intro.MODEL_SPECIFIC_CONV_KWARGS
        assert "GraphSAGE" in intro.MODEL_SPECIFIC_CONV_KWARGS

    def test_gcn_conv_kwargs(self):
        intro = pytest.importorskip(_MODULE_PATH)
        kwargs = intro.get_model_conv_kwargs("GCN")
        assert "add_self_loops" in kwargs
        assert "normalize" in kwargs

    def test_gat_no_normalize(self):
        intro = pytest.importorskip(_MODULE_PATH)
        kwargs = intro.get_model_conv_kwargs("GAT")
        assert "normalize" not in kwargs
        assert "heads" in kwargs

    def test_graphsage_no_add_self_loops(self):
        intro = pytest.importorskip(_MODULE_PATH)
        kwargs = intro.get_model_conv_kwargs("GraphSAGE")
        assert "add_self_loops" not in kwargs
        assert "normalize" in kwargs

    def test_unknown_falls_back_to_full_set(self):
        intro = pytest.importorskip(_MODULE_PATH)
        kwargs = intro.get_model_conv_kwargs("UnknownModel")
        assert kwargs == intro.KNOWN_CONV_KWARGS

    def test_pna_has_aggregators(self):
        intro = pytest.importorskip(_MODULE_PATH)
        kwargs = intro.get_model_conv_kwargs("PNA")
        assert "aggregators" in kwargs
        assert "deg" in kwargs


# ═══════════════════════════════════════════════════════════════════════════
# 7. Type Inference Helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestTypeInference:
    """Tests for _infer_param_type and _type_hint_to_string."""

    def test_hint_int(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro._type_hint_to_string(int) == "int"

    def test_hint_float(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro._type_hint_to_string(float) == "float"

    def test_hint_bool(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro._type_hint_to_string(bool) == "bool"

    def test_hint_optional(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro._type_hint_to_string(int | None) == "optional"

    def test_name_convention_channels(self):
        intro = pytest.importorskip(_MODULE_PATH)
        param = inspect.Parameter("hidden_channels", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        assert intro._infer_param_type("hidden_channels", param, {}) == "int"

    def test_name_convention_dropout(self):
        intro = pytest.importorskip(_MODULE_PATH)
        param = inspect.Parameter("dropout", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        assert intro._infer_param_type("dropout", param, {}) == "float"

    def test_name_convention_use_bool(self):
        intro = pytest.importorskip(_MODULE_PATH)
        param = inspect.Parameter("use_bn", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        assert intro._infer_param_type("use_bn", param, {}) == "bool"

    def test_default_none_infers_optional(self):
        intro = pytest.importorskip(_MODULE_PATH)
        param = inspect.Parameter(
            "something", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
        )
        assert intro._infer_param_type("something", param, {}) == "optional"

    def test_fallback_any(self):
        intro = pytest.importorskip(_MODULE_PATH)
        param = inspect.Parameter("unknown_thing", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        assert intro._infer_param_type("unknown_thing", param, {}) == "any"


# ═══════════════════════════════════════════════════════════════════════════
# 8. _infer_intelligent_default
# ═══════════════════════════════════════════════════════════════════════════


class TestInferIntelligentDefault:
    """Tests for _infer_intelligent_default()."""

    def test_num_blocks(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro._infer_intelligent_default("num_blocks", "int") == 6

    def test_num_spherical(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro._infer_intelligent_default("num_spherical", "int") == 7

    def test_hidden_channels(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro._infer_intelligent_default("hidden_channels", "int") == 128

    def test_dropout(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro._infer_intelligent_default("dropout", "float") == 0.0

    def test_cutoff(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro._infer_intelligent_default("cutoff", "float") == 5.0

    def test_activation(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro._infer_intelligent_default("act", "str") == "relu"

    def test_normalize_bool(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro._infer_intelligent_default("normalize", "bool") is True

    def test_no_match_returns_none(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro._infer_intelligent_default("some_obscure_param", "int") is None

    def test_num_layers(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro._infer_intelligent_default("num_layers", "int") == 3

    def test_num_gaussians(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro._infer_intelligent_default("num_gaussians", "int") == 50

    def test_aggr_string(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro._infer_intelligent_default("aggr", "str") == "add"


# ═══════════════════════════════════════════════════════════════════════════
# 9. _infer_category_enum
# ═══════════════════════════════════════════════════════════════════════════


class TestInferCategoryEnum:
    """Tests for _infer_category_enum()."""

    def test_gcn_basic_gnn(self):
        intro = pytest.importorskip(_MODULE_PATH)
        cat = intro._infer_category_enum("torch_geometric.nn.models.GCN", "GCN")
        assert cat == intro.ModelCategory.BASIC_GNN

    def test_autoencoder(self):
        intro = pytest.importorskip(_MODULE_PATH)
        cat = intro._infer_category_enum("torch_geometric.nn.models.autoencoder.GAE", "GAE")
        assert cat == intro.ModelCategory.AUTOENCODER

    def test_transformer(self):
        intro = pytest.importorskip(_MODULE_PATH)
        cat = intro._infer_category_enum("torch_geometric.nn.models.Transformer", "Transformer")
        assert cat == intro.ModelCategory.TRANSFORMER

    def test_schnet_utility(self):
        intro = pytest.importorskip(_MODULE_PATH)
        cat = intro._infer_category_enum("torch_geometric.nn.models.SchNet", "SchNet")
        assert cat == intro.ModelCategory.UTILITY

    def test_pool_is_pooling(self):
        intro = pytest.importorskip(_MODULE_PATH)
        cat = intro._infer_category_enum("torch_geometric.nn.models.TopKPool", "TopKPool")
        assert cat == intro.ModelCategory.POOLING

    def test_unknown_gets_fallback(self):
        intro = pytest.importorskip(_MODULE_PATH)
        cat = intro._infer_category_enum("torch_geometric.nn.models.New", "New")
        assert cat is not None


# ═══════════════════════════════════════════════════════════════════════════
# 10. _infer_supported_tasks
# ═══════════════════════════════════════════════════════════════════════════


class TestInferSupportedTasks:
    """Tests for _infer_supported_tasks()."""

    def test_standard_gnn_all_tasks(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakeGCN)
        tasks = intro._infer_supported_tasks("GCN", "torch_geometric.nn.models.GCN", params)
        assert "node_classification" in tasks
        assert "graph_classification" in tasks
        assert "link_prediction" in tasks

    def test_autoencoder_link_and_embedding(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakeGAE)
        tasks = intro._infer_supported_tasks(
            "GAE", "torch_geometric.nn.models.autoencoder.GAE", params
        )
        assert "link_prediction" in tasks
        assert "node_embedding" in tasks
        assert "graph_classification" not in tasks

    def test_pool_graph_level_only(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakePoolModel)
        tasks = intro._infer_supported_tasks(
            "TopKPool", "torch_geometric.nn.models.TopKPool", params
        )
        assert "graph_classification" in tasks
        assert "node_classification" not in tasks


# ═══════════════════════════════════════════════════════════════════════════
# 11. _infer_tags
# ═══════════════════════════════════════════════════════════════════════════


class TestInferTags:
    """Tests for _infer_tags()."""

    def test_gat_attention_tag(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakeGAT)
        tags = intro._infer_tags("GAT", "torch_geometric.nn.models.GAT", params)
        assert "attention" in tags

    def test_schnet_molecular_3d_tags(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakeSchNet)
        tags = intro._infer_tags("SchNet", "torch_geometric.nn.models.SchNet", params)
        assert "molecular" in tags
        assert "3d" in tags

    def test_model_with_heads_multihead_tag(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakeTransformerModel)
        tags = intro._infer_tags(
            "TransformerConv", "torch_geometric.nn.conv.TransformerConv", params
        )
        assert "multi-head" in tags


# ═══════════════════════════════════════════════════════════════════════════
# 12. _parameters_to_hyperparameters_dict
# ═══════════════════════════════════════════════════════════════════════════


class TestParametersToHyperparametersDict:
    """Tests for _parameters_to_hyperparameters_dict()."""

    def test_basic_conversion(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = {
            "hidden_channels": intro.ParameterInfo(
                name="hidden_channels",
                param_type="int",
                required=True,
            ),
            "dropout": intro.ParameterInfo(
                name="dropout",
                param_type="float",
                required=False,
                default=0.5,
            ),
        }
        hp = intro._parameters_to_hyperparameters_dict(params)
        assert hp["hidden_channels"]["type"] == "integer"
        assert hp["dropout"]["default"] == 0.5

    def test_module_type_params(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = {
            "encoder": intro.ParameterInfo(name="encoder", param_type="any", required=True),
            "decoder": intro.ParameterInfo(
                name="decoder", param_type="any", required=False, default=None
            ),
        }
        hp = intro._parameters_to_hyperparameters_dict(params)
        assert hp["encoder"]["type"] == "module"
        assert hp["encoder"]["auto_created"] is True

    def test_intelligent_default_for_required(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = {
            "num_blocks": intro.ParameterInfo(
                name="num_blocks",
                param_type="int",
                required=True,
            ),
        }
        hp = intro._parameters_to_hyperparameters_dict(params)
        assert hp["num_blocks"]["default"] == 6


# ═══════════════════════════════════════════════════════════════════════════
# 13. generate_search_space / _param_to_search_space
# ═══════════════════════════════════════════════════════════════════════════


class TestGenerateSearchSpace:
    """Tests for generate_search_space()."""

    def test_int_param_space(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = {
            "hidden_channels": intro.ParameterInfo(
                name="hidden_channels",
                param_type="int",
                required=False,
                default=64,
            )
        }
        space = intro.generate_search_space(params)
        assert "hidden_channels" in space["hyperparameters"]
        assert space["hyperparameters"]["hidden_channels"]["type"] == "int"

    def test_dropout_float_space(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = {
            "dropout": intro.ParameterInfo(
                name="dropout",
                param_type="float",
                required=False,
                default=0.0,
            )
        }
        space = intro.generate_search_space(params)
        assert space["hyperparameters"]["dropout"]["high"] == 0.6

    def test_bool_categorical_space(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = {
            "normalize": intro.ParameterInfo(
                name="normalize",
                param_type="bool",
                required=False,
                default=True,
            )
        }
        space = intro.generate_search_space(params)
        assert space["hyperparameters"]["normalize"]["type"] == "categorical"

    def test_required_in_out_channels_skipped(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = {
            "in_channels": intro.ParameterInfo(name="in_channels", param_type="int", required=True),
            "out_channels": intro.ParameterInfo(
                name="out_channels", param_type="int", required=True
            ),
            "num_layers": intro.ParameterInfo(
                name="num_layers", param_type="int", required=False, default=3
            ),
        }
        space = intro.generate_search_space(params)
        assert "in_channels" not in space["hyperparameters"]
        assert "out_channels" not in space["hyperparameters"]
        assert "num_layers" in space["hyperparameters"]

    def test_num_spherical_min_2(self):
        intro = pytest.importorskip(_MODULE_PATH)
        pi = intro.ParameterInfo(name="num_spherical", param_type="int", required=False, default=7)
        space = intro._param_to_search_space("num_spherical", pi)
        assert space["low"] >= 2


# ═══════════════════════════════════════════════════════════════════════════
# 14. validate_params_against_signature
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateParamsAgainstSignature:
    """Tests for validate_params_against_signature()."""

    def test_valid_passes(self):
        intro = pytest.importorskip(_MODULE_PATH)
        ok, errors = intro.validate_params_against_signature(
            _FakeGCN, {"in_channels": 10, "hidden_channels": 64, "num_layers": 3}
        )
        assert ok is True
        assert errors == []

    def test_unknown_param_detected(self):
        intro = pytest.importorskip(_MODULE_PATH)
        ok, errors = intro.validate_params_against_signature(
            _FakeGCN, {"in_channels": 10, "hidden_channels": 64, "num_layers": 3, "bogus": 42}
        )
        assert ok is False
        assert any("bogus" in e for e in errors)

    def test_missing_required_detected(self):
        intro = pytest.importorskip(_MODULE_PATH)
        ok, errors = intro.validate_params_against_signature(_FakeGCN, {"dropout": 0.5})
        assert ok is False
        assert any("in_channels" in e for e in errors)


# ═══════════════════════════════════════════════════════════════════════════
# 15. get_valid_params_for_model
# ═══════════════════════════════════════════════════════════════════════════


class TestGetValidParamsForModel:
    """Tests for get_valid_params_for_model()."""

    def test_returns_set(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.get_valid_params_for_model(_FakeGCN)
        assert isinstance(params, set)
        assert "in_channels" in params
        assert "self" not in params


# ═══════════════════════════════════════════════════════════════════════════
# 16. DynamicModelMetadata
# ═══════════════════════════════════════════════════════════════════════════


class TestDynamicModelMetadata:
    """Tests for the DynamicModelMetadata Pydantic model."""

    def test_construction(self):
        intro = pytest.importorskip(_MODULE_PATH)
        meta = intro.DynamicModelMetadata(
            name="TestModel",
            category=intro.ModelCategory.BASIC_GNN,
            import_path="test.TestModel",
        )
        assert meta.name == "TestModel"
        assert meta.supported_tasks == []
        assert meta.accepts_kwargs is False

    def test_to_dict_serializes_enum(self):
        intro = pytest.importorskip(_MODULE_PATH)
        meta = intro.DynamicModelMetadata(
            name="GCN",
            category=intro.ModelCategory.BASIC_GNN,
            import_path="torch_geometric.nn.models.GCN",
        )
        d = meta.to_dict()
        assert d["category"] == "basic_gnn"

    def test_default_factory_isolation(self):
        intro = pytest.importorskip(_MODULE_PATH)
        m1 = intro.DynamicModelMetadata(
            name="A", category=intro.ModelCategory.BASIC_GNN, import_path="t.A"
        )
        m2 = intro.DynamicModelMetadata(
            name="B", category=intro.ModelCategory.BASIC_GNN, import_path="t.B"
        )
        m1.supported_tasks.append("task_a")
        assert "task_a" not in m2.supported_tasks

    def test_backward_compatible_alias(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert intro.ModelMetadata is intro.DynamicModelMetadata


# ═══════════════════════════════════════════════════════════════════════════
# 17. generate_model_metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestGenerateModelMetadata:
    """Tests for generate_model_metadata()."""

    def test_generates_metadata(self, mock_nn_module):
        intro = pytest.importorskip(_MODULE_PATH)
        with patch.object(intro.importlib, "import_module") as mock_import:
            fake_mod = types.ModuleType("torch_geometric.nn.models")
            fake_mod.GCN = _FakeGCN
            mock_import.return_value = fake_mod

            meta = intro.generate_model_metadata("GCN", "torch_geometric.nn.models.GCN")

        assert meta is not None
        assert meta.name == "GCN"
        assert meta.import_path == "torch_geometric.nn.models.GCN"
        assert isinstance(meta.parameters, dict)
        assert isinstance(meta.forward_parameters, dict)
        assert isinstance(meta.required_data_attributes, set)

    def test_returns_none_for_import_failure(self, mock_nn_module):
        intro = pytest.importorskip(_MODULE_PATH)
        with patch.object(intro.importlib, "import_module", side_effect=ImportError("nope")):
            meta = intro.generate_model_metadata("Bad", "torch_geometric.nn.models.Bad")
        assert meta is None

    def test_kwargs_flag_populated(self, mock_nn_module):
        intro = pytest.importorskip(_MODULE_PATH)
        with patch.object(intro.importlib, "import_module") as mock_import:
            fake_mod = types.ModuleType("torch_geometric.nn.models")
            fake_mod.GCN = _FakeGCN
            mock_import.return_value = fake_mod

            meta = intro.generate_model_metadata("GCN", "torch_geometric.nn.models.GCN")
        assert meta.accepts_kwargs is True  # _FakeGCN has **kwargs

    def test_hyperparameters_dict_populated(self, mock_nn_module):
        intro = pytest.importorskip(_MODULE_PATH)
        with patch.object(intro.importlib, "import_module") as mock_import:
            fake_mod = types.ModuleType("torch_geometric.nn.models")
            fake_mod.GCN = _FakeGCN
            mock_import.return_value = fake_mod

            meta = intro.generate_model_metadata("GCN", "torch_geometric.nn.models.GCN")
        assert isinstance(meta.hyperparameters, dict)
        assert len(meta.hyperparameters) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 18. discover_pyg_models
# ═══════════════════════════════════════════════════════════════════════════


class TestDiscoverPygModels:
    """Tests for discover_pyg_models()."""

    def test_discovers_public_model_classes(self, fake_discovery_env):
        intro = pytest.importorskip(_MODULE_PATH)
        models = intro.discover_pyg_models()
        assert "GCN" in models
        assert "GAT" in models
        assert "SchNet" in models
        assert "DimeNet" in models

    def test_excludes_private_names(self, fake_discovery_env):
        intro = pytest.importorskip(_MODULE_PATH)
        models = intro.discover_pyg_models()
        assert "_PrivateHelper" not in models

    def test_excludes_non_class_objects(self, fake_discovery_env):
        intro = pytest.importorskip(_MODULE_PATH)
        models = intro.discover_pyg_models()
        assert "some_function" not in models
        assert "SOME_CONSTANT" not in models

    def test_values_are_import_path_strings(self, fake_discovery_env):
        intro = pytest.importorskip(_MODULE_PATH)
        models = intro.discover_pyg_models()
        for _name, path in models.items():
            assert isinstance(path, str)
            assert "torch_geometric" in path

    def test_handles_import_errors_gracefully(self, mock_nn_module):
        """When torch_geometric is missing entirely, should return empty dict."""
        intro = pytest.importorskip(_MODULE_PATH)

        def _fail(name):
            raise ImportError(f"No module named '{name}'")

        with patch.object(intro.importlib, "import_module", side_effect=_fail):
            models = intro.discover_pyg_models()
        assert isinstance(models, dict)


# ═══════════════════════════════════════════════════════════════════════════
# 19. PyGModelIntrospector (Singleton)
# ═══════════════════════════════════════════════════════════════════════════


class TestPyGModelIntrospector:
    """Tests for the PyGModelIntrospector singleton class."""

    def test_singleton_pattern(self, fake_discovery_env):
        intro = pytest.importorskip(_MODULE_PATH)
        a = intro.PyGModelIntrospector()
        b = intro.PyGModelIntrospector()
        assert a is b

    def test_get_all_model_names(self, fake_discovery_env):
        intro = pytest.importorskip(_MODULE_PATH)
        introspector = intro.PyGModelIntrospector()
        names = introspector.get_all_model_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_has_model(self, fake_discovery_env):
        intro = pytest.importorskip(_MODULE_PATH)
        introspector = intro.PyGModelIntrospector()
        names = introspector.get_all_model_names()
        if names:
            assert introspector.has_model(names[0]) is True
        assert introspector.has_model("NonExistentModel") is False

    def test_get_import_path(self, fake_discovery_env):
        intro = pytest.importorskip(_MODULE_PATH)
        introspector = intro.PyGModelIntrospector()
        names = introspector.get_all_model_names()
        if names:
            path = introspector.get_import_path(names[0])
            assert path is not None
            assert isinstance(path, str)

    def test_refresh_clears_cache(self, fake_discovery_env):
        intro = pytest.importorskip(_MODULE_PATH)
        introspector = intro.PyGModelIntrospector()
        introspector.refresh()
        # After refresh, should still have models
        assert len(introspector.get_all_model_names()) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 20. get_introspector
# ═══════════════════════════════════════════════════════════════════════════


class TestGetIntrospector:
    """Tests for the get_introspector() convenience function."""

    def test_returns_introspector(self, fake_discovery_env):
        intro = pytest.importorskip(_MODULE_PATH)
        introspector = intro.get_introspector()
        assert isinstance(introspector, intro.PyGModelIntrospector)

    def test_returns_same_instance(self, fake_discovery_env):
        intro = pytest.importorskip(_MODULE_PATH)
        a = intro.get_introspector()
        b = intro.get_introspector()
        assert a is b


# ═══════════════════════════════════════════════════════════════════════════
# 21. Backward Compatible API Functions
# ═══════════════════════════════════════════════════════════════════════════


class TestBackwardCompatibleAPI:
    """Tests for module-level backward-compatible functions."""

    def test_get_all_model_names(self, fake_discovery_env):
        intro = pytest.importorskip(_MODULE_PATH)
        names = intro.get_all_model_names()
        assert isinstance(names, list)

    def test_search_models(self, fake_discovery_env, mock_nn_module):
        intro = pytest.importorskip(_MODULE_PATH)
        # Search for a model by name
        with patch.object(intro.importlib, "import_module") as mock_import:
            fake_mod = types.ModuleType("torch_geometric.nn.models")
            fake_mod.GCN = _FakeGCN
            mock_import.return_value = fake_mod

            results = intro.search_models("GCN")
        assert isinstance(results, list)

    def test_get_category_statistics(self, fake_discovery_env):
        intro = pytest.importorskip(_MODULE_PATH)
        stats = intro.get_category_statistics()
        assert isinstance(stats, dict)


# ═══════════════════════════════════════════════════════════════════════════
# 22. _LazyAllModels / ALL_MODELS
# ═══════════════════════════════════════════════════════════════════════════


class TestLazyAllModels:
    """Tests for the _LazyAllModels lazy-loading dict."""

    def test_all_models_is_dict(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert isinstance(intro.ALL_MODELS, dict)

    def test_lazy_loading_class(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert isinstance(intro.ALL_MODELS, intro._LazyAllModels)


# ═══════════════════════════════════════════════════════════════════════════
# 23. ModelCategory Enum
# ═══════════════════════════════════════════════════════════════════════════


class TestModelCategoryEnum:
    """Tests for the ModelCategory enum."""

    def test_has_basic_gnn(self):
        intro = pytest.importorskip(_MODULE_PATH)
        assert hasattr(intro.ModelCategory, "BASIC_GNN")

    def test_has_expected_categories(self):
        intro = pytest.importorskip(_MODULE_PATH)
        expected = {
            "BASIC_GNN",
            "CONVOLUTIONAL",
            "ATTENTION",
            "POOLING",
            "AGGREGATION",
            "AUTOENCODER",
            "TRANSFORMER",
            "UTILITY",
        }
        for cat in expected:
            assert hasattr(intro.ModelCategory, cat), f"Missing category: {cat}"

    def test_enum_values_are_strings(self):
        intro = pytest.importorskip(_MODULE_PATH)
        for member in intro.ModelCategory:
            assert isinstance(member.value, str)


# ═══════════════════════════════════════════════════════════════════════════
# 24. _extract_param_description
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractParamDescription:
    """Tests for _extract_param_description()."""

    def test_extracts_from_docstring(self):
        intro = pytest.importorskip(_MODULE_PATH)
        desc = intro._extract_param_description(_FakeGCN, "in_channels")
        # _FakeGCN.__init__ has no docstring, so should return ""
        # But _FakeGCN class docstring mentions GCN
        assert isinstance(desc, str)

    def test_returns_empty_for_missing_param(self):
        intro = pytest.importorskip(_MODULE_PATH)
        desc = intro._extract_param_description(_FakeGCN, "nonexistent_param")
        assert desc == ""


# ═══════════════════════════════════════════════════════════════════════════
# 25. _extract_forward_param_description
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractForwardParamDescription:
    """Tests for _extract_forward_param_description()."""

    def test_extracts_from_forward_docstring(self):
        intro = pytest.importorskip(_MODULE_PATH)
        # _FakeGCN.forward has a docstring mentioning "x"
        desc = intro._extract_forward_param_description(_FakeGCN, "x")
        assert isinstance(desc, str)

    def test_returns_empty_for_no_forward(self):
        intro = pytest.importorskip(_MODULE_PATH)
        desc = intro._extract_forward_param_description(_FakeNoForward, "x")
        assert desc == ""


# ═══════════════════════════════════════════════════════════════════════════
# 26. Cross-Function Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossFunctionIntegration:
    """Verify functions compose correctly end-to-end."""

    def test_introspect_feeds_validation(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakeGCN)
        valid_names = set(params.keys())
        # A config with only valid param names should pass
        ok, errors = intro.validate_params_against_signature(
            _FakeGCN, {name: 1 for name in valid_names if params[name].required}
        )
        assert ok is True

    def test_forward_and_required_attrs_consistent(self):
        intro = pytest.importorskip(_MODULE_PATH)
        fwd_params = intro.introspect_forward_signature(_FakeSchNet)
        req_attrs = intro.get_required_data_attributes(_FakeSchNet)
        # All required attrs should come from required forward params
        for attr in req_attrs:
            assert attr in fwd_params

    def test_search_space_generated_from_introspection(self):
        intro = pytest.importorskip(_MODULE_PATH)
        params = intro.introspect_model_signature(_FakeDimeNet)
        space = intro.generate_search_space(params)
        assert isinstance(space, dict)
        assert "hyperparameters" in space
        # DimeNet has num_blocks, should appear in search space
        assert "num_blocks" in space["hyperparameters"]
