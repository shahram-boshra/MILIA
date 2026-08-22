# Transformation Plugins

This is the **transformation** plugin kind-container, symmetric with `../descriptors/` and
`../models/`. Every first-party and user transformation plugin lives here, one category per
sub-directory.

## Layout

```
transformations/
├── __init__.py                 # kind-container package
├── README.md
├── <category>/                 # a transform plugin (e.g. graph_products, curvature_rewiring)
│   ├── __init__.py             # registers the category's transform classes
│   ├── plugin.yaml             # manifest (semver + version ranges + declarations)
│   └── transforms.py           # the transform implementations
└── user_template/              # copy this to start a new transform plugin
    ├── __init__.py.template
    ├── plugin.yaml.template
    ├── transforms.py.template
    └── README.md
```

## Discovery

The transform `PluginRegistry` scans this directory for `*/plugin.yaml` (one level). The default
in-tree search path resolves package-relative via
`milia_pipeline.plugins.get_transform_plugins_directory()` (CWD-independent) and is set in
`configs/plugins.yaml` (`plugins.plugin_paths`). Additional locations — including user-owned
directories outside the package tree — can be appended to `plugins.plugin_paths`.

## Adding a transform plugin

Copy `user_template/` to a new `<your_category>/`, rename the `.template` files, and implement your
transforms (subclass `CustomTransformBase`, declare them in `plugin.yaml`, register them in
`__init__.py`). See `user_template/README.md`.
