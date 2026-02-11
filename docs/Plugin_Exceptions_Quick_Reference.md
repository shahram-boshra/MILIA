# Plugin Exceptions Quick Reference Guide

**Module:** `exceptions.py`  
**Phase:** 3 - Step 3.2.2  
**Date:** 2025-10-12

---

## Quick Import

```python
from vqm24_pipeline.exceptions import (
    PluginError,                    # Base exception
    PluginValidationError,          # Validation failures
    PluginSecurityError,            # Security issues
    PluginDependencyError,          # Missing dependencies
    PluginDiscoveryError,           # Discovery problems
    PluginRegistrationError,        # Registration conflicts
    PluginLoadError                 # Loading failures
)
```

---

## Exception Hierarchy

```
BaseProjectError
    └── PluginError
            ├── PluginValidationError
            ├── PluginSecurityError
            ├── PluginDependencyError
            ├── PluginDiscoveryError
            ├── PluginRegistrationError
            └── PluginLoadError
```

---

## When to Use Each Exception

### PluginError
**Use for:** General plugin system errors that don't fit other categories
```python
raise PluginError(
    "Unexpected plugin system error",
    plugin_name="my_plugin",
    details="Additional context"
)
```

### PluginValidationError
**Use for:** Plugin validation failures (dependencies, transforms, parameters, compatibility, security)
```python
raise PluginValidationError(
    "Plugin failed validation",
    plugin_name="my_plugin",
    validation_errors=[
        "Missing dependency: torch>=1.9.0",
        "Transform instantiation failed"
    ]
)
```

### PluginSecurityError
**Use for:** Security concerns (unsafe code, dangerous imports, checksum failures)
```python
raise PluginSecurityError(
    "Security issues detected",
    plugin_name="untrusted_plugin",
    security_issues=[
        "Uses subprocess module",
        "Contains eval() calls"
    ]
)
```

### PluginDependencyError
**Use for:** Missing or incompatible dependencies
```python
raise PluginDependencyError(
    "Dependencies not satisfied",
    plugin_name="my_plugin",
    missing_dependencies=[
        "torch>=1.9.0",
        "numpy>=1.20.0"
    ]
)
```

### PluginDiscoveryError
**Use for:** Problems discovering or parsing plugins
```python
raise PluginDiscoveryError(
    "Failed to discover plugin",
    plugin_name="broken_plugin",
    discovery_path="/path/to/plugins",
    details="plugin.yaml is malformed"
)
```

### PluginRegistrationError
**Use for:** Plugin registration conflicts or failures
```python
raise PluginRegistrationError(
    "Plugin already registered",
    plugin_name="new_plugin",
    conflicting_plugin="existing_plugin"
)
```

### PluginLoadError
**Use for:** Module import or initialization failures
```python
raise PluginLoadError(
    "Failed to load plugin module",
    plugin_name="broken_plugin",
    load_path="/path/to/plugin.py",
    original_error="ImportError: No module named 'missing_dep'"
)
```

---

## Error Handling Patterns

### Pattern 1: Try-Catch Specific Exception
```python
try:
    plugin_registry.discover_plugins(paths)
except PluginDiscoveryError as e:
    logger.error(f"Discovery failed: {e}")
    logger.debug(f"Discovery path: {e.discovery_path}")
```

### Pattern 2: Catch All Plugin Errors
```python
try:
    plugin_registry.validate_plugin(plugin_name)
except PluginError as e:
    logger.error(f"Plugin error: {e}")
    logger.debug(f"Plugin: {e.plugin_name}")
```

### Pattern 3: Chain Exceptions
```python
try:
    module = import_plugin_module(path)
except ImportError as e:
    raise PluginLoadError(
        "Module import failed",
        plugin_name=plugin_name,
        load_path=str(path),
        original_error=str(e)
    ) from e
```

### Pattern 4: Conditional Handling
```python
try:
    plugin_registry.validate_plugin(plugin_name)
except PluginValidationError as e:
    if e.validation_errors:
        for error in e.validation_errors:
            logger.warning(f"Validation error: {error}")
    # Continue with next plugin
except PluginSecurityError as e:
    # Security errors are critical
    logger.critical(f"Security issue: {e}")
    raise  # Re-raise
```

---

## Best Practices

### 1. Always Include Context
✅ Good:
```python
raise PluginError(
    "Plugin initialization failed",
    plugin_name=plugin_name,
    details=f"Failed at step: {current_step}"
)
```

❌ Bad:
```python
raise PluginError("Failed")
```

### 2. Use Appropriate Exception Type
✅ Good:
```python
if missing_deps:
    raise PluginDependencyError(
        "Dependencies not satisfied",
        plugin_name=name,
        missing_dependencies=missing_deps
    )
```

❌ Bad:
```python
if missing_deps:
    raise PluginError("Dependencies missing")
```

### 3. Preserve Original Errors
✅ Good:
```python
try:
    load_module(path)
except ImportError as e:
    raise PluginLoadError(...) from e
```

❌ Bad:
```python
try:
    load_module(path)
except ImportError as e:
    raise PluginLoadError(...)
```

### 4. Log Before Raising Critical Errors
```python
if security_issues:
    logger.critical(f"Security issues in {plugin_name}: {security_issues}")
    raise PluginSecurityError(
        "Critical security issues detected",
        plugin_name=plugin_name,
        security_issues=security_issues
    )
```

---

## Common Scenarios

### Scenario 1: Plugin Discovery
```python
def discover_plugins(self, paths: List[Path]) -> List[PluginMetadata]:
    discovered = []
    
    for path in paths:
        try:
            metadata = self._load_plugin_metadata(path)
            discovered.append(metadata)
        except PluginDiscoveryError as e:
            logger.warning(f"Skipping {path}: {e}")
            continue  # Try next plugin
        except Exception as e:
            raise PluginDiscoveryError(
                "Unexpected error during discovery",
                discovery_path=str(path),
                details=str(e)
            ) from e
    
    return discovered
```

### Scenario 2: Plugin Validation
```python
def validate_plugin(self, plugin_name: str) -> Dict[str, Any]:
    metadata = self.get_plugin_info(plugin_name)
    
    # Check dependencies
    missing = self._check_dependencies(metadata)
    if missing:
        raise PluginDependencyError(
            "Plugin dependencies not satisfied",
            plugin_name=plugin_name,
            missing_dependencies=missing
        )
    
    # Check security
    issues = self._check_security(metadata)
    if issues:
        raise PluginSecurityError(
            "Security issues detected",
            plugin_name=plugin_name,
            security_issues=issues
        )
    
    return {"passed": True}
```

### Scenario 3: Plugin Loading
```python
def load_plugin(self, plugin_name: str, plugin_path: Path) -> ModuleType:
    try:
        spec = importlib.util.spec_from_file_location(
            plugin_name, 
            plugin_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
        
    except ImportError as e:
        raise PluginLoadError(
            "Failed to import plugin module",
            plugin_name=plugin_name,
            load_path=str(plugin_path),
            original_error=str(e),
            details="Check plugin dependencies"
        ) from e
        
    except Exception as e:
        raise PluginLoadError(
            "Unexpected error loading plugin",
            plugin_name=plugin_name,
            load_path=str(plugin_path),
            original_error=str(e)
        ) from e
```

---

## Testing Examples

### Test 1: Verify Exception Attributes
```python
def test_plugin_error_attributes():
    exc = PluginError(
        "Test error",
        plugin_name="test_plugin",
        details="Test details"
    )
    
    assert exc.plugin_name == "test_plugin"
    assert exc.message == "Test error"
    assert exc.details == "Test details"
```

### Test 2: Verify Exception Message
```python
def test_plugin_validation_error_message():
    exc = PluginValidationError(
        "Validation failed",
        plugin_name="test_plugin",
        validation_errors=["Error 1", "Error 2"]
    )
    
    message = str(exc)
    assert "test_plugin" in message
    assert "Error 1" in message
    assert "Error 2" in message
```

### Test 3: Verify Exception Inheritance
```python
def test_plugin_exception_hierarchy():
    assert issubclass(PluginValidationError, PluginError)
    assert issubclass(PluginError, BaseProjectError)
```

---

## CLI Integration Example

```python
def handle_plugin_validation_command(args):
    try:
        results = PluginRegistry.validate_plugin(args.plugin_name)
        print(f"✓ Validation passed for {args.plugin_name}")
        return 0
        
    except PluginDependencyError as e:
        print(f"✗ Missing dependencies:")
        for dep in e.missing_dependencies:
            print(f"  - {dep}")
        return 1
        
    except PluginSecurityError as e:
        print(f"✗ Security issues detected:")
        for issue in e.security_issues:
            print(f"  - {issue}")
        return 2
        
    except PluginValidationError as e:
        print(f"✗ Validation failed:")
        for error in e.validation_errors:
            print(f"  - {error}")
        return 3
        
    except PluginError as e:
        print(f"✗ Plugin error: {e}")
        return 4
```

---

## Troubleshooting

### Issue: Exception not caught
**Solution:** Check exception hierarchy - use `PluginError` to catch all plugin exceptions

### Issue: Missing error context
**Solution:** Always include `plugin_name` and `details` parameters

### Issue: Original error lost
**Solution:** Use `raise ... from e` to preserve exception chain

### Issue: Unclear error messages
**Solution:** Check `__str__()` output - use specific exception types

---

## Additional Resources

- **Full Documentation:** See `Phase3_Step3.2.2_Implementation_Summary.md`
- **Exception Module:** `exceptions.py`
- **Plugin System:** `plugin_system.py` (Sub-Step 3.2.1)
- **Blueprint:** `VQM24_refactoring_07_01_new_modules_ph03_Blueprint_32.txt`

---

**Quick Reference Version:** 1.0  
**Last Updated:** 2025-10-12
