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
│   ├── plugin.yaml             # manifest (declares model class_name + module_path)
│   └── model.py                # the model implementation
└── user_template/              # copy this to start a new model plugin
    ├── plugin.yaml.template
    ├── model.py.template
    └── README.md
```

## Discovery

The model plugin loader scans this directory for `*/plugin.yaml` (one level). The default
search path is resolved package-relative via
`milia_pipeline.plugins.get_model_plugins_directory()` (CWD-independent). Additional
locations — including user-owned directories outside the package tree — can be registered
through the `plugins.plugin_paths` configuration key.

## Adding a model plugin

Copy `user_template/` to a new `<your_category>/` directory, rename the `.template` files,
and fill in the manifest and model class. See `user_template/README.md` for details.
