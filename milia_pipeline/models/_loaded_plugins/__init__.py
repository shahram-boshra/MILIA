"""Namespace for the modules of model plugins loaded from directories.

The model plugin loader imports each directory plugin as the subpackage
``milia_pipeline.models._loaded_plugins.<plugin_name>``, whose search path is the
plugin's directory. A plugin's ``module_path: model`` is therefore imported as
``milia_pipeline.models._loaded_plugins.<plugin_name>.model``. Plugins never
share module names, and ``sys.path`` is not changed.

Nothing in this package is public API; it holds no code of its own.
"""
