# Model Plugin — User Template

Scaffold for adding your own **model** plugin, symmetric with
`../../descriptors/user_template/`.

## Steps

1. Copy this `user_template/` directory to a new one named for your plugin, e.g.
   `milia_pipeline/plugins/models/my_model/` (or any directory you register via the
   `plugins.plugin_paths` config key — it need not live inside the package tree).
2. Rename the template files:
   - `plugin.yaml.template` → `plugin.yaml`
   - `model.py.template` → `model.py`
3. Fill in `plugin.yaml` (manifest schema version 2: metadata, `requirements` and the
   `models:` declarations). Each declaration's `class` is `"<module>:<Class>"`, e.g.
   `"model:MyModel"` for the class `MyModel` in `model.py`. Unknown keys are rejected, and
   each error names its field. `category` must be a category of MILIA's model registry;
   omit `supported_tasks` to infer them from the class. A manifest without
   `schema_version` is the deprecated legacy format (`class_name` + `module_path`,
   `milia_version`, …): it still loads, with a deprecation warning per legacy key, until
   MILIA 2.0.0.
4. Implement your model class in `model.py` (a `torch.nn.Module`). Import MILIA and
   third-party packages with absolute imports; import other modules of your plugin with
   relative imports (`from .helpers import build_mlp`). The loader imports the plugin
   directory as its own package and does not add it to `sys.path`, so `import helpers`
   would not find a sibling module.
   `plugin_name` must be a valid Python identifier (e.g. `my_models`).
5. The loader discovers the plugin by scanning for `plugin.yaml` (one level) under each
   configured model plugin path.

## Notes

- The default in-tree model plugin path resolves package-relative via
  `milia_pipeline.plugins.get_model_plugins_directory()` (CWD-independent).
- Keep the plugin self-contained; declare third-party needs under `dependencies` as PEP 508
  requirements on distribution names (e.g. `"torch>=2.4"`, `"scikit-learn"`, not the import name
  `sklearn`). They are checked against the installed packages; nothing is imported.
