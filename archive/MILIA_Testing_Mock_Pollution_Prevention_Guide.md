# MILIA Pipeline: Mock Pollution Prevention Guide

**Project:** MILIA Pipeline  
**Version:** 1.0.0  
**Date:** 2026-02-11  
**Evidence Base:** 8 polluter files hardened, 1,132 tests verified, 22,575 full-suite collection with 0 errors  
**Companion Document:** `MILIA_Test_Infrastructure_Tracker.md` §4.3–§4.4  

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

# Step 2 — Full suite collection (must remain at 22,575+ tests, 0 errors):
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
