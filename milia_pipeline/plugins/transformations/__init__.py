"""
Transformation plugins kind-container.

This package groups all **transformation** plugin categories (graph products, combinatorial/
dual operators, structural padding, augmentations, spectral signals, graph reduction, curvature
rewiring, topological lifts, latent-structural encodings, PyG-parity augmentations, ...). It is
symmetric with ``../descriptors/`` and ``../models/`` — one kind-container per plugin kind.

Each immediate sub-directory is one transform plugin category holding a ``plugin.yaml`` manifest
plus a ``transforms`` module and an ``__init__.py`` that registers its transform classes. The
transform ``PluginRegistry`` discovers plugins by scanning this directory for ``*/plugin.yaml``
(one level); the default search path is resolved package-relative via
``milia_pipeline.plugins.get_transform_plugins_directory()`` and is configurable through the
``plugins.plugin_paths`` key.

To add a transform plugin, copy ``user_template/`` to a new ``<category>/`` directory (or register
an external directory via ``plugins.plugin_paths``). See ``user_template/README.md``.
"""
