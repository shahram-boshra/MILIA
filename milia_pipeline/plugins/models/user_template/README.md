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
3. Fill in `plugin.yaml` (metadata + `models:` declarations). Ensure each declaration's
   `class_name` matches the class in `model.py` and `module_path` is `model`.
4. Implement your model class in `model.py` (a `torch.nn.Module`). Use absolute imports.
5. The loader discovers the plugin by scanning for `plugin.yaml` (one level) under each
   configured model plugin path.

## Notes

- The default in-tree model plugin path resolves package-relative via
  `milia_pipeline.plugins.get_model_plugins_directory()` (CWD-independent).
- Keep the plugin self-contained; declare third-party needs under `dependencies`.
