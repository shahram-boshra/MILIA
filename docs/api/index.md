# API Reference

This section documents the public API of the `milia_pipeline` package.

## Core Modules

| Module | Description |
|--------|-------------|
| `milia_pipeline.config` | Configuration management, validation, and schema definitions |
| `milia_pipeline.molecules` | Molecular processing, conversion, and feature enrichment |
| `milia_pipeline.transformations` | Graph transformation system with plugin support |
| `milia_pipeline.datasets` | PyTorch Geometric dataset implementations |
| `milia_pipeline.handlers` | Unified dataset handler pattern |
| `milia_pipeline.preprocessing` | Wavefunction data preprocessing |
| `milia_pipeline.descriptors` | Molecular descriptor calculation (3,000+ descriptors: ~492 built-in + plugins, e.g. addcore_3d's 1,019 + constitutional_property's 20 + topological_connectivity's 107 + walk_path_information's 86 + matrix_spectral's 177 + autocorrelation_2d's 606 + estate_atomtype's 316 + cats2d's 150 + eta's 45 + cpsa_geometric_3d's 50 + eht_electronic's 25) |
| `milia_pipeline.models` | Model training, evaluation, and deployment |
| `milia_pipeline.plugins` | Plugin system for extensibility |
| `milia_pipeline.cli_manager` | Command-line interface |
| `milia_pipeline.exceptions` | Three-tier exception hierarchy |

## Public API and compatibility

MILIA follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Its public API is:

- every module listed in the table above, and their submodules whose names do not start with `_`;
- in each of those modules, the names listed in its `__all__`.

Everything else is internal and may change in any release: modules and names starting with `_`, the
test suite, and anything not reachable by the rules above.

A public name or module that is going away is first deprecated: it keeps working, emits a
`DeprecationWarning` naming its replacement and the release that removes it, and is listed under
**Deprecated** in the changelog. It is removed only in a major release.

### Deprecated import paths

| Deprecated | Use instead | Removal |
|------------|-------------|---------|
| `milia_pipeline.models.plugins` | `milia_pipeline.models.model_plugin_system` | 2.0.0 |
| `milia_pipeline.models.plugins.model_plugin_system` | `milia_pipeline.models.model_plugin_system` | 2.0.0 |
