"""``PluginModelError`` is both a model error and a plugin error.

Handlers written for plugin errors (``except PluginError``) and for model errors
(``except ModelError``) both catch it, and every constructor argument survives
the cooperative ``super().__init__`` chain:
``PluginModelError -> ModelError -> PluginError -> BaseProjectError``.
"""

from __future__ import annotations

import pickle

import pytest

from milia_pipeline.exceptions import (
    BaseProjectError,
    ModelError,
    PluginError,
    PluginModelError,
)


def test_every_argument_survives_the_init_chain():
    error = PluginModelError(
        "load failed",
        plugin_name="my_plugin",
        model_name="CustomGNN",
        details="module not found",
        plugin_path="/plugins/my_plugin",
    )

    assert error.message == "load failed"
    assert error.plugin_name == "my_plugin"
    assert error.model_name == "CustomGNN"
    assert error.details == "module not found"
    assert error.plugin_path == "/plugins/my_plugin"
    assert error.extra_info == {}


def test_positional_arguments_keep_their_meaning():
    error = PluginModelError("load failed", "my_plugin", "CustomGNN", "/plugins/my_plugin")

    assert (error.plugin_name, error.model_name, error.plugin_path) == (
        "my_plugin",
        "CustomGNN",
        "/plugins/my_plugin",
    )
    assert error.details is None


def test_extra_keyword_arguments_reach_extra_info():
    error = PluginModelError("load failed", plugin_name="my_plugin", attempt=2)

    assert error.extra_info == {"attempt": 2}
    assert error.plugin_name == "my_plugin"


def test_is_a_plugin_error_and_a_model_error_with_model_error_first():
    assert issubclass(PluginModelError, ModelError)
    assert issubclass(PluginModelError, PluginError)
    assert issubclass(PluginModelError, BaseProjectError)
    mro = PluginModelError.__mro__
    assert mro.index(ModelError) < mro.index(PluginError) < mro.index(BaseProjectError)


@pytest.mark.parametrize("handler", [PluginError, ModelError, BaseProjectError])
def test_caught_by_plugin_and_model_error_handlers(handler):
    with pytest.raises(handler):
        raise PluginModelError("load failed", plugin_name="my_plugin")


def test_message_names_the_plugin_like_other_plugin_errors():
    error = PluginModelError("load failed", plugin_name="my_plugin", details="module not found")
    plugin_error = PluginError("load failed", plugin_name="my_plugin", details="module not found")

    assert str(error) == "load failed (Plugin: my_plugin). Details: module not found"
    assert str(error) == str(plugin_error)
    assert str(PluginModelError("load failed")) == "load failed"


def test_survives_pickling():
    error = pickle.loads(pickle.dumps(PluginModelError("load failed", plugin_name="my_plugin")))

    assert isinstance(error, PluginModelError)
    assert error.message == "load failed"


def test_model_error_keeps_details():
    """Regression: ModelError passes ``details`` on by keyword."""
    error = ModelError("bad model", model_name="GCN", details="shape mismatch")

    assert error.details == "shape mismatch"
    assert error.model_name == "GCN"
    assert str(error) == "bad model. Details: shape mismatch"
