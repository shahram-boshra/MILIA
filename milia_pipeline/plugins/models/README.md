# Model Plugins

This is the **model** plugin kind-container, symmetric with `../transformations/` and
`../descriptors/`. First-party and user model plugins live here, each in its own category
sub-directory holding a `plugin.yaml` manifest plus the module that implements the model
class.

## Layout

```
models/
├── README.md
├── <category>/                 # a first-party or user model plugin
│   ├── plugin.yaml             # manifest, schema version 2 (declares each model's class)
│   └── model.py                # the model implementation
├── example_models/             # first-party example: ExampleGNN
└── user_template/              # copy this to start a new model plugin
    ├── plugin.yaml.template
    ├── model.py.template
    └── README.md
```

## Discovery

Plugins are discovered and loaded automatically, once per process, when the first
`ModelFactory` is created (and before HPO filters a search space), while
`models.plugins.enabled` and `auto_discover` are `true`. Their models are then available by
name like any other model. `ensure_model_plugins_discovered()` does the same explicitly.

The model plugin loader scans this directory for `*/plugin.yaml` (one level). It is found
package-relative via `milia_pipeline.plugins.get_model_plugins_directory()` (independent of the
working directory) and scanned while `models.plugins.include_builtin` is `true` (the default).
Your own directories, including ones outside the package tree, go in
`models.plugins.plugin_paths`: absolute, `~/...`, or relative to the configuration root (the
`configs/` directory, or the directory of a single configuration file). A relative path that
exists only relative to the working directory still works, with a deprecation warning, until
MILIA 2.0.0. A path that is not an existing directory is logged and skipped.

## Using a plugin model

A registered plugin model is used by its declared `name`, exactly like a PyG model, in both
model-selection modes and for every task type (node, graph and edge regression and
classification, link prediction). The factory adds what the task needs: pooling per graph for
graph-level tasks, an edge decoder for edge tasks and link prediction.

```yaml
# configs/models.yaml
models:
  selection:
    mode: "single"
    model_name: "ExampleGNN"
```

```yaml
models:
  selection:
    mode: "ensemble"
    model_name: "ensemble"
  ensemble:
    strategy: "hierarchical"        # or "parallel", "sequential"
    models:
      - {name: "GCN", hyperparameters: {}, level: 0}
      - {name: "ExampleGNN", hyperparameters: {}, level: 1}
```

An ensemble may combine PyG and plugin models in any proportion. Sequential and hierarchical
ensembles feed each member the previous member's output, so their members must be chainable
(see the model conventions in `user_template/README.md`). The deprecated `custom` mode
(removal in 2.0.0) is replaced by a model plugin: write the architecture as a plugin model and
use it by name.

## Distributing a model plugin as a package

An installed distribution can provide a model plugin through the `milia.models` entry-point
group. The entry point names a package that holds `plugin.yaml`, and its name must equal the
manifest's `plugin_name`:

```toml
# pyproject.toml of the distribution
[project.entry-points."milia.models"]
my_plugin = "my_dist.milia_plugin"
```

```yaml
# my_dist/milia_plugin/plugin.yaml
schema_version: 2
plugin_name: my_plugin
# ... the other fields as in user_template/plugin.yaml.template
models:
  - name: MyNet
    class: "my_dist.milia_plugin.model:MyNet"   # absolute import path
```

Include `plugin.yaml` in the distribution as package data. Entry-point plugins are discovered
with the directories (directories first, so a directory plugin keeps a name both use) while
`models.plugins.discover_entry_points` is `true` (the default); discovery imports the package,
so install only plugins you trust. A failing entry point is logged and does not stop the others.

## Validation

Each plugin is checked before registration; the log gives the reason for every plugin that is
skipped or rejected.

- **Compatibility:** the installed MILIA, Python and PyG versions must satisfy `requirements`
  (PEP 440 specifiers), and the MILIA extra named by `requires_extra` must be installed;
  otherwise the plugin is skipped.
- **Errors, at every level:** a manifest that does not match schema version 2 (or the legacy
  format), a missing or unsatisfied `dependencies` entry (PEP 508), or an incomplete model
  declaration. A plugin with errors is not registered.
- **Warnings:** a missing version or no models. `models.plugins.validation_level: strict` also
  rejects a plugin with warnings; `standard` (the default) and `permissive` register it.
- **Code scan:** at `standard` and `strict` the plugin's Python files are scanned for dynamic
  execution (`eval(`, `exec(`, `__import__(`, `os.system(`, `import subprocess`). Findings are
  logged at INFO as hints for whoever reviews the plugin and never affect registration.

## Trust model

Loading a plugin imports its modules, which runs their code with the privileges of the MILIA
process. Plugins are therefore trusted code: a directory plugin is code you put on a plugin
path, an entry-point plugin is a distribution you installed. MILIA neither sandboxes nor vets
plugin code, and the code scan above is not a security control. Install and enable only
plugins you would run as your own code; to turn sources off, set `include_builtin`,
`discover_entry_points` or `models.plugins.enabled` to `false`.

## Adding a model plugin

Copy `user_template/` to a new `<your_category>/` directory, rename the `.template` files,
and fill in the manifest and model class. See `user_template/README.md` for details.
