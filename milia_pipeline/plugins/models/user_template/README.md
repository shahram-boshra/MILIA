# Model Plugin — User Template

Scaffold for adding your own **model** plugin, symmetric with
`../../descriptors/user_template/`.

## Steps

1. Copy this `user_template/` directory to a new one named for your plugin, e.g.
   `milia_pipeline/plugins/models/my_model/`, or a new directory inside one you list in
   `models.plugins.plugin_paths` (absolute, `~/...`, or relative to `configs/`; it need not
   live inside the package tree).
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
4. Implement your model class in `model.py` (a `torch.nn.Module`) following the conventions
   below. Import MILIA and third-party packages with absolute imports; import other
   modules of your plugin with relative imports (`from .helpers import build_mlp`). The
   loader imports the plugin directory as its own package and does not add it to
   `sys.path`, so `import helpers` would not find a sibling module.
   `plugin_name` must be a valid Python identifier (e.g. `my_models`).
5. The loader discovers the plugin by scanning for `plugin.yaml` (one level) under each
   configured model plugin path, when the first `ModelFactory` is created.

`../example_models/` is a complete, working plugin to compare against.

## Model conventions

- Constructor parameters `in_channels`, `hidden_channels` and `out_channels`; set
  `self.in_channels` and `self.out_channels`.
- `forward(x, edge_index, batch=None)` (`edge_attr` may follow `edge_index`), or
  `forward(z, pos, batch=None)` for 3D models, returning one row per node
  (`[num_nodes, out_channels]`). For graph-level tasks the factory pools the rows per graph.

A model that does not follow them is still registered, with a warning:

- `forward() does not follow (x, edge_index[, edge_attr][, batch]) or (z, pos[, batch]);
  registered; not composable in ensembles`: it can be built on its own, not in ensembles.
- `constructor lacks in_channels and out_channels/hidden_channels; registered; not chainable
  in sequential or hierarchical ensembles`: it can join parallel ensembles only, because
  sequential and hierarchical ensembles set each member's `in_channels` from the previous
  member's output size.

## Notes

- The built-in container (`milia_pipeline/plugins/models`) is found package-relative via
  `milia_pipeline.plugins.get_model_plugins_directory()` (independent of the working
  directory) and scanned while `models.plugins.include_builtin` is `true`.
- Keep the plugin self-contained; declare third-party needs under `dependencies` as PEP 508
  requirements on distribution names (e.g. `"torch>=2.4"`, `"scikit-learn"`, not the import name
  `sklearn`). They are checked against the installed packages; nothing is imported.
- A plugin is registered only if it is compatible and valid. When the installed MILIA, Python
  or PyG version does not satisfy `requirements`, or the MILIA extra named by `requires_extra`
  is not installed, the plugin is skipped and the log gives the reason. A plugin whose
  validation reports errors (for example a missing dependency) is not registered at any
  `validation_level`; `strict` also rejects warnings (for example a missing version).
