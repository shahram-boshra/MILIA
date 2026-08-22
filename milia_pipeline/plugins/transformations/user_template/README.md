# Transform Plugin — User Template

Scaffold for adding your own **transformation** plugin, symmetric with
`../../descriptors/user_template/`.

## Steps

1. Copy this `user_template/` directory to a new one named for your category, e.g.
   `milia_pipeline/plugins/transformations/my_transforms/` (or any directory you register via the
   `plugins.plugin_paths` config key — it need not live inside the package tree).
2. Rename the template files:
   - `__init__.py.template` → `__init__.py`
   - `plugin.yaml.template` → `plugin.yaml`
   - `transforms.py.template` → `transforms.py`
3. Implement your transforms in `transforms.py` (subclass `CustomTransformBase`;
   `transform(data) -> data` purity contract; deterministic unless explicitly seeded).
4. Declare each transform in `plugin.yaml` (`class_name` + `module_path: transforms`) and list the
   class names in `__init__.py`'s `_transform_names`.
5. Discovery: the transform `PluginRegistry` scans each configured transform plugin path for
   `*/plugin.yaml` (one level). The default in-tree path resolves package-relative via
   `milia_pipeline.plugins.get_transform_plugins_directory()`.

## Notes

- Keep the plugin self-contained; declare third-party needs under `dependencies` (prefer none).
- Use absolute imports so the module loads regardless of the plugin's location.
