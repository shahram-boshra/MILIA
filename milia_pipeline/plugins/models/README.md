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
└── user_template/              # copy this to start a new model plugin
    ├── plugin.yaml.template
    ├── model.py.template
    └── README.md
```

## Discovery

The model plugin loader scans this directory for `*/plugin.yaml` (one level). It is found
package-relative via `milia_pipeline.plugins.get_model_plugins_directory()` (independent of the
working directory) and scanned while `models.plugins.include_builtin` is `true` (the default).
Your own directories, including ones outside the package tree, go in
`models.plugins.plugin_paths`: absolute, `~/...`, or relative to the configuration root (the
`configs/` directory, or the directory of a single configuration file). A relative path that
exists only relative to the working directory still works, with a deprecation warning, until
MILIA 2.0.0. A path that is not an existing directory is logged and skipped.

## Adding a model plugin

Copy `user_template/` to a new `<your_category>/` directory, rename the `.template` files,
and fill in the manifest and model class. See `user_template/README.md` for details.
