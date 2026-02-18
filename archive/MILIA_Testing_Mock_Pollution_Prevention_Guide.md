# MILIA Pipeline: Mock Pollution Prevention Guide

**Project:** MILIA Pipeline
**Version:** 2.0.0
**Date:** 2026-02-18
**Evidence Base:** 8 polluter files hardened, 1,132 tests verified, 22,601 full-suite collection with 0 errors; 18 additional failures fixed (8 pre-existing + 10 pollution)
**Companion Document:** `MILIA_Test_Infrastructure_Tracker.md` §4.3–§4.4, `TEST_SUITE_POLLUTION_TRACKER.md`

---

## Core Problem

Injecting mocks into `sys.modules` at **module level** (top-level code outside any function) pollutes the global import system during **pytest collection**. `teardown_module()` does NOT run during collection ([pytest docs](https://docs.pytest.org/en/stable/how-to/xunit_setup.html)), so mocks persist and break subsequent test files. This caused 17 collection errors across the MILIA test suite, resolved by hardening 8 polluter files.

---

## Rule 1: NEVER Write to `sys.modules` at Module Level

Any line at module level (not inside a function or class) that writes to `sys.modules` is a **polluter**. This includes:
- `sys.modules['X'] = mock_obj` (direct injection)
- `for name in mocks: sys.modules[name] = mocks[name]` (injection loops)
- `del sys.modules['X']` (deletion — also mutates global state)
- `importlib.util` loading followed by `sys.modules['X'] = loaded_module`

**Allowed at module level** (zero `sys.modules` side-effects):
- Mock class definitions (`class MockX: ...`)
- Mock object construction (`mock_obj = Mock()`)
- Dict construction (`_mock_modules = {'X': mock_obj}`)
- Standard library imports (`import pytest`, `from pathlib import Path`)

---

## Rule 2: Defer ALL `sys.modules` Writes to `setup_module()`

Use this exact pattern (proven across 8 MILIA test files, 1,132 tests):

```python
import sys
from pathlib import Path

# §4.2: Dynamic project root — NEVER hardcode paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---- SAFE at module level: pure memory, zero sys.modules writes ----
_mock_modules = {
    'some.module': Mock(),
    'another.module': Mock(),
}
_original_modules = {}

# ---- Placeholders for names extracted from the module-under-test ----
SomeClass = None
some_function = None

def setup_module(module):
    """Inject mocks + import module-under-test. Runs ONCE before tests."""
    global SomeClass, some_function

    # 1. Inject mocks into sys.modules
    for name in _mock_modules:
        if name in sys.modules:
            _original_modules[name] = sys.modules[name]
        sys.modules[name] = _mock_modules[name]

    # 2. Import the module-under-test (NOW safe — mocks are in place)
    from milia_pipeline.some_module import SomeClass as _SC, some_function as _SF
    SomeClass = _SC
    some_function = _SF

    # 3. Publish into module namespace (ensures test fixtures can see them)
    module.SomeClass = SomeClass
    module.some_function = some_function

def teardown_module(module):
    """Restore sys.modules to pre-test state."""
    for name in _mock_modules:
        if name in sys.modules:
            if name in _original_modules:
                sys.modules[name] = _original_modules[name]
            else:
                del sys.modules[name]
```

---

## Rule 3: `@patch` Cannot Replace `setup_module()` for Import-Time Mocking

`@patch` and context managers operate at *test execution time*. They **cannot** mock dependencies that the module-under-test imports at *its own top level*. For example, if `config_loader.py` does `from milia_pipeline.handlers.config_handler import ConfigHandler` at line 1, that import runs when `config_loader` is first loaded — before any `@patch` decorator activates. The `setup_module()` pattern is the correct solution for this MILIA-specific pattern.

`@patch` is still appropriate for mocking objects *within* individual test methods (e.g., patching a function call inside a test), but not for `sys.modules` injection.

---

## Rule 4: Verify with AST — Not Just grep

`grep` finds the text `sys.modules[` but cannot distinguish module-level writes (polluters) from writes inside functions (safe). The definitive check is AST analysis:

```bash
python3 -c "
import ast, sys
with open('tests/test_FILENAME.py') as f:
    source = f.read()
tree = ast.parse(source)
lines = source.splitlines()
found = False
for node in ast.iter_child_nodes(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        continue
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        continue
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        continue
    dump = ast.dump(node)
    if 'sys' in dump and 'modules' in dump:
        print(f'POLLUTER Line {node.lineno}: {lines[node.lineno-1].strip()}')
        found = True
print('CLEAN' if not found else 'POLLUTED — must defer to setup_module()')
"
```

If the output says `CLEAN`, the file is safe. If `POLLUTED`, apply the Rule 2 fix pattern.

---

## Rule 5: Three-Step Verification After Every New Test File

```bash
# Step 1 — Isolation collection (no errors):
pytest tests/test_NEW_FILE.py --collect-only 2>&1 | tail -5

# Step 2 — Full suite collection (must remain at 22,601+ tests, 0 errors):
pytest tests/ --collect-only 2>&1 | tail -5

# Step 3 — Isolation run (all pass):
pytest tests/test_NEW_FILE.py -v --tb=short 2>&1 | tail -10
```

Step 2 is **critical** — a file can pass Steps 1 and 3 while still polluting `sys.modules` and breaking other files.

---

## Rule 6: NEVER Hardcode Paths

Use the §4.2 dynamic pattern in every test file:
```python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
```
Never use `Path('/app/milia')`, `Path(__file__).parent.parent.absolute()`, or any string literal path.

---

## Rule 7: `mock.patch()` Must Target Where the Name Is Looked Up

Python's `mock.patch()` replaces an attribute on a **module object**. The patch target must be the module where the name is resolved at runtime — not necessarily where the function is defined.

### Module-Level Imports

When the module under test imports at the top of its file:
```python
# source_module.py
def my_function(): ...

# consumer_module.py
from source_module import my_function   # <-- module-level import
```
Patch at the **consumer**: `@patch("consumer_module.my_function")` — because `consumer_module.__dict__["my_function"]` is where the name resolves.

### Local Imports (Inside Function Bodies)

When the module under test imports inside a function:
```python
# consumer_module.py
def some_method(self):
    from source_module import my_function   # <-- local import
    my_function()
```
Patch at the **source**: `@patch("source_module.my_function")` — because `from source_module import my_function` re-fetches from `sys.modules["source_module"]` at each call. Patching the consumer has no effect because the name never exists in the consumer's `__dict__`.

### How to Determine Which Case Applies

1. Open the module under test.
2. Search for the function name in the **top-level imports** (outside any function/class body).
3. If found → patch at the consumer (`module_under_test.function_name`).
4. If NOT found → check if it's imported locally inside a function body → patch at the source (`source_module.function_name`).
5. If NOT found anywhere → the attribute doesn't exist on the module; the `@patch` is invalid and will raise `AttributeError`.

### Common Errors (from MILIA test suite — 8 failures fixed)

| Error | Root Cause | Fix |
|-------|-----------|-----|
| `AttributeError: module 'milia_dataset' does not have attribute 'get_combined_transforms_as_dicts'` | Function only imported locally inside method bodies, not at module level | Patch at source: `config_accessors.get_combined_transforms_as_dicts` |
| `AttributeError: module 'nas_manager' does not have attribute 'SearchSpaceError'` | Exception class never imported by `nas_manager.py` (only appears in docstrings) | Patch at source: `milia_pipeline.exceptions.SearchSpaceError` |
| `AttributeError: module 'qdpi' does not have attribute 'HAR2EV'` | Constant never imported by `qdpi.py`; used only in `milia_dataset.py` | Patch at source: `config_constants.HAR2EV` |
| `AttributeError: module 'molecule_converter_core' does not have attribute 'load_config'` | Function never imported by `molecule_converter_core.py` at all | Remove the invalid patch entirely |

---

## Rule 8: Execution-Time Pollution Categories and Fixes

Beyond collection-time pollution (Rules 1–3), tests can corrupt state **during execution**. Three categories exist:

### Category B — Registry Singleton Cleared

A test calls `.clear()` or `.reset()` on a global singleton registry (dataset, descriptor, or model) for isolation, but does not restore contents. Subsequent tests find empty registries.

**Symptoms**: `list_all()` returns `[]`, `is_registered("DFT")` returns `False`, handler tests get empty required-property lists.

**Fix**: Tests that call `.clear()` must save and restore the registry dict in a `try/finally` block. The `conftest.py` hook provides a safety net (see Rule 9).

### Category C — Class Identity Mismatch After Module Reload

A test reloads a module (or deletes it from `sys.modules`, causing reimport). The new module creates NEW class objects. `isinstance()` and `issubclass()` checks fail because `old_module.MyClass is not new_module.MyClass`.

**Symptoms**: `except ConfigurationError:` doesn't catch, `isinstance(handler, DatasetHandler)` returns `False`, `issubclass` checks fail silently.

**Fix**: Protect critical modules via `conftest.py` `_CRITICAL_PREFIXES` (see Rule 9). The hook restores original module objects after each test.

### Category D — Stale `func.__globals__` After Module Reload

Functions imported at the top of a test file retain `__globals__` bound to the OLD module's `__dict__` after reload. `mock.patch()` patches the NEW module's `__dict__`, but the function resolves names from the OLD dict. Mocks become invisible.

**Symptoms**: `mock.patch` appears to have no effect; functions use real implementations instead of mocks; tests pass solo but fail in suite.

**Fix**: Either protect the module via `conftest.py` `_CRITICAL_PREFIXES`, or re-fetch functions from `sys.modules` in `setUp()`:
```python
def setUp(self):
    mod = sys.modules["milia_pipeline.some_module"]
    self.my_function = getattr(mod, "my_function")
```

---

## Rule 9: `conftest.py` Module and Registry Protection

The `tests/conftest.py` provides automatic protection against execution-time pollution via two hooks:

### How It Works

1. **`pytest_collection_finish`**: After collection (all imports done), snapshots:
   - `sys.modules` entries for critical module prefixes (original module objects)
   - Registry singleton contents (dataset, descriptor, model dicts)

2. **`pytest_runtest_teardown` (hookwrapper, trylast)**: After EACH test's complete teardown:
   - Restores any deleted/reloaded modules in `sys.modules`
   - Restores any cleared registry contents from snapshots

### Protected Module Prefixes (`_CRITICAL_PREFIXES`)

```python
_CRITICAL_PREFIXES = (
    "milia_pipeline.exceptions",
    "milia_pipeline.datasets.registry",
    "milia_pipeline.datasets.base",
    "milia_pipeline.datasets.implementations",
    "milia_pipeline.descriptors",
    "milia_pipeline.models.registry",
    "milia_pipeline.transformations.plugin_system",
    "milia_pipeline.transformations.custom_transforms",
    "milia_pipeline.transformations.graph_transforms",
    "milia_pipeline.molecules.mol_conversion_utils",
)
```

### When to Add a New Prefix

Add a module to `_CRITICAL_PREFIXES` when:
1. Tests in a file pass solo but fail in the full suite (pollution symptom)
2. The failing tests use `mock.patch()` targeting attributes on that module
3. The module is NOT manipulated via `setup_module()`/`teardown_module()` by any test file

**Do NOT add a prefix** if test files legitimately use `setup_module()` to inject mocks into that module's `sys.modules` entry. Protecting such modules prevents the legitimate mocks from taking effect. This is why `milia_pipeline.handlers` and `milia_pipeline.config` are NOT in the protected list — `test_config_loader_unit.py` uses `setup_module()` injection for these.

### Registry Snapshots

| Registry | Module Path | Singleton Access | Internal Dict |
|----------|------------|-----------------|---------------|
| Dataset | `milia_pipeline.datasets.registry` | `_default_registry` | `._datasets` |
| Descriptor | `milia_pipeline.descriptors.descriptor_registry` | `DescriptorRegistry._instances[cls]` | `._descriptors`, `._by_category` |
| Model | `milia_pipeline.models.registry.model_registry` | `ModelRegistry._instances[cls]` | `._models` |

---

## Rule 10: Four-Step Verification After Every Test Fix

Extends Rule 5 with a full-suite execution check:

```bash
# Step 1 — Isolation collection (no errors):
pytest tests/test_NEW_FILE.py --collect-only 2>&1 | tail -5

# Step 2 — Full suite collection (must remain at 22,601+ tests, 0 errors):
pytest tests/ --collect-only 2>&1 | tail -5

# Step 3 — Isolation run (all pass):
pytest tests/test_NEW_FILE.py -v --tb=short 2>&1 | tail -10

# Step 4 — Full suite run for specific tests (pollution check):
pytest tests/ -k "test_name_1 or test_name_2" -v --tb=short 2>&1 | tail -15
```

Step 4 is **critical for pollution fixes** — a test can pass in Steps 1–3 while still failing when preceded by a polluter in the full suite. Only Step 4 catches this.

---

## Evidence: Files Hardened Using This Pattern

| # | File | Tests | Pattern Applied |
|---|------|-------|-----------------|
| 1 | `test_architecture_builder_unit.py` | 105 | Injection loop + real imports → `setup_module()` |
| 2 | `test_config_loader_unit.py` | 153 | 3 injections + 35 dependent imports → `setup_module()` |
| 3 | `test_config_parser_unit.py` | 114 | Injection loop + dependent imports → `setup_module()` |
| 4 | `test_layer_registry_unit.py` | 108 | Injection loop + `importlib.util` loading → `setup_module()` |
| 5 | `test_model_composer_integration.py` | 19 | Injection loop → `setup_module()` |
| 6 | `test_model_composer_unit.py` | 158 | Injection loop → `setup_module()` |
| 7 | `test_model_plugin_system_unit.py` | 107 | Injection loop → `setup_module()` |
| 9 | `test_templates_unit.py` | 130 | Injection loop + `importlib.util` loading → `setup_module()` |
| 10 | `test_validation_unit.py` | 71 | 6 injections + `importlib.util` loading → `setup_module()` |
| 11 | `test_validators_unit.py` | 165 | 9 injections + `importlib` loading + Phase 6 extraction → `setup_module()` |
| — | `test_preprocessing_init_unit.py` | 2 | **False positive** — `sys.modules` writes already inside function body + autouse fixture isolation |
| **Total** | | **1,132** | |

### Phase 2 Fixes (2026-02-18): Pre-Existing + Pollution

| # | File | Tests Fixed | Fix Applied |
|---|------|------------|-------------|
| 1 | `test_milia_dataset_unit.py` | 3 | Corrected `mock.patch()` paths: local imports → patch at source module (Rule 7) |
| 2 | `test_hpo_nas_nas_manager.py` | 3 | Corrected `mock.patch()` paths: attributes not imported by consumer (Rule 7) |
| 3 | `test_handler_impl_qdpi_unit.py` | 1 | Corrected `mock.patch()` path: `HAR2EV` not imported by `qdpi.py` (Rule 7) |
| 4 | `test_e2e_preprocessing_workflow.py` | 1 | Removed invalid `mock.patch()`: `load_config` never imported by target module (Rule 7) |
| 5 | `tests/conftest.py` | 10 | Added 4 modules to `_CRITICAL_PREFIXES` for Category C/D protection (Rule 9) |
| **Total** | | **18** | |
