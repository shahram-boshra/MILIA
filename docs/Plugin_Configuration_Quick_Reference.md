# Plugin Configuration Quick Reference
## VQM24 Pipeline - Phase 3.2.5

**Last Updated:** October 13, 2025  
**For:** VQM24 Users and Developers

---

## 🚀 Quick Start

### Enable Plugins (Minimal Configuration)

Add to your `config.yaml`:

```yaml
plugins:
  enabled: true
  plugin_paths:
    - ./plugins
```

That's it! All other settings will use sensible defaults.

---

## 📋 All Configuration Options

```yaml
plugins:
  # Core Settings
  enabled: false                      # Enable/disable plugin system
  plugin_paths:                       # Where to find plugins
    - ./plugins
  
  # Discovery & Validation
  auto_discover: true                 # Auto-find plugins on startup
  auto_validate: true                 # Auto-validate plugins
  validation_level: "standard"        # strict/standard/permissive/disabled
  
  # Whitelisting & Blacklisting
  trusted_plugins: []                 # Pre-approved plugins (skip validation)
  disabled_plugins: []                # Never load these plugins
  
  # Feature Flags
  allow_experimental: false           # Allow beta/experimental plugins
  max_plugins: 50                     # Maximum plugins to load
  require_metadata: true              # Require plugin.yaml file
  
  # Security
  enforce_checksums: false            # Verify plugin file integrity
  security_scanning: false            # Scan for security issues
  
  # Loading Behavior
  fail_on_plugin_error: false         # Stop if plugin fails to load
  cache_validation_results: true      # Cache for faster loading
  
  # Integration
  allow_override_builtin: false       # Allow plugins to override built-in transforms
  warn_on_name_conflict: true         # Warn if plugin names conflict
```

---

## 🎯 Common Scenarios

### Scenario 1: Development

**Goal:** Quick iteration, test new plugins

```yaml
plugins:
  enabled: true
  plugin_paths:
    - ./plugins
    - ./dev_plugins
  validation_level: "permissive"
  allow_experimental: true
  allow_override_builtin: true
  fail_on_plugin_error: false
```

### Scenario 2: Production

**Goal:** Maximum security and stability

```yaml
plugins:
  enabled: true
  plugin_paths:
    - /opt/vqm24/plugins
  validation_level: "strict"
  trusted_plugins:
    - official_molecular_transforms
  allow_experimental: false
  enforce_checksums: true
  security_scanning: true
  fail_on_plugin_error: true
```

### Scenario 3: Testing Specific Plugin

**Goal:** Test one plugin, disable others

```yaml
plugins:
  enabled: true
  plugin_paths:
    - ./plugins
  disabled_plugins:
    - old_plugin_v1
    - buggy_transform
  validation_level: "standard"
```

### Scenario 4: High-Performance

**Goal:** Fast loading, skip validation

```yaml
plugins:
  enabled: true
  plugin_paths:
    - ./plugins
  validation_level: "permissive"
  auto_validate: false
  trusted_plugins:
    - my_fast_transform
  cache_validation_results: true
```

---

## 🔐 Security Levels

### Level 1: Permissive (Development Only)

```yaml
validation_level: "permissive"
enforce_checksums: false
security_scanning: false
allow_experimental: true
```

**Use when:**
- Developing plugins locally
- Rapid prototyping
- Trusted environment only

**Risk:** ⚠️ High - minimal safety checks

### Level 2: Standard (Recommended Default)

```yaml
validation_level: "standard"
enforce_checksums: false
security_scanning: false
allow_experimental: false
```

**Use when:**
- Normal development
- Testing with trusted plugins
- Most production scenarios

**Risk:** ✅ Low - balanced safety and performance

### Level 3: Strict (High Security)

```yaml
validation_level: "strict"
enforce_checksums: true
security_scanning: true
allow_experimental: false
trusted_plugins: [...]
fail_on_plugin_error: true
```

**Use when:**
- Production deployments
- Security-sensitive data
- Regulated industries
- Reproducible research

**Risk:** ✅ Minimal - maximum safety checks

---

## 🔧 Using Plugin Transforms

### In Experimental Setups

```yaml
plugins:
  enabled: true
  plugin_paths:
    - ./plugins

transformations:
  experimental_setups:
    my_custom_setup:
      name: "my_custom_setup"
      description: "Using plugin transforms"
      enabled: true
      transforms:
        # Built-in transform
        - name: "AddSelfLoops"
          enabled: true
        
        # Plugin transform (same syntax!)
        - name: "MyCustomTransform"  # From plugin
          enabled: true
          params:
            my_param: 42
  
  default_setup: "my_custom_setup"
```

**Key Point:** Plugin transforms use the exact same syntax as built-in transforms!

---

## 🎨 CLI Override Examples

### Override Disabled Plugins

**config.yaml:**
```yaml
plugins:
  enabled: false
```

**CLI:**
```bash
python main.py --enable-plugin my_plugin
```

**Result:** Plugin system enabled, `my_plugin` loaded

### Add Plugin Path

**config.yaml:**
```yaml
plugins:
  enabled: true
  plugin_paths:
    - ./plugins
```

**CLI:**
```bash
python main.py --plugin-path /opt/custom_plugins
```

**Result:** Both `./plugins` and `/opt/custom_plugins` searched

### Discover and List

```bash
python main.py --discover-plugins --list-plugins
```

**Result:** Finds all plugins and displays list

### Validate Plugin

```bash
python main.py --validate-plugin my_transform
```

**Result:** Runs validation checks on specific plugin

---

## 📁 Directory Structure

### Recommended Layout

```
project_root/
├── config.yaml          # Your configuration
├── plugins/             # Default plugin directory
│   ├── plugin1/
│   │   ├── __init__.py
│   │   ├── plugin.yaml  # Required metadata
│   │   └── transforms.py
│   └── plugin2/
│       ├── __init__.py
│       ├── plugin.yaml
│       └── custom.py
└── vqm24_pipeline/      # VQM24 code
```

### Plugin Paths

**Absolute paths:**
```yaml
plugin_paths:
  - /opt/vqm24/plugins
  - /home/user/my_plugins
```

**Relative paths (from project root):**
```yaml
plugin_paths:
  - ./plugins
  - ../shared_plugins
```

**Environment variables:**
```yaml
plugin_paths:
  - $HOME/.vqm24/plugins
  - $VQM24_PLUGIN_DIR
```

---

## ⚠️ Common Mistakes

### ❌ Mistake 1: Forgetting to Enable

```yaml
plugins:
  plugin_paths:
    - ./plugins
  # Missing: enabled: true
```

**Fix:**
```yaml
plugins:
  enabled: true
  plugin_paths:
    - ./plugins
```

### ❌ Mistake 2: Wrong Validation Level

```yaml
plugins:
  validation_level: "super_strict"  # Invalid!
```

**Fix:** Use one of: `"strict"`, `"standard"`, `"permissive"`, `"disabled"`

```yaml
plugins:
  validation_level: "strict"
```

### ❌ Mistake 3: Non-Existent Path

```yaml
plugins:
  plugin_paths:
    - /does/not/exist
```

**Fix:** Ensure path exists or will log warning

```yaml
plugins:
  plugin_paths:
    - ./plugins  # Make sure this directory exists
```

### ❌ Mistake 4: Conflicting Settings

```yaml
plugins:
  validation_level: "strict"
  allow_experimental: true  # Conflicts with strict!
```

**Fix:** Use consistent security levels

```yaml
plugins:
  validation_level: "strict"
  allow_experimental: false
```

---

## 🔍 Troubleshooting

### Issue: Plugins Not Loading

**Symptoms:** Plugin transforms not found

**Check:**
1. `enabled: true` set?
2. Plugin path exists?
3. Plugin has `plugin.yaml`?
4. Plugin in `disabled_plugins` list?

**Solution:**
```bash
# List discovered plugins
python main.py --list-plugins

# Validate specific plugin
python main.py --validate-plugin my_plugin
```

### Issue: Validation Failures

**Symptoms:** Plugin fails validation checks

**Check:**
1. Validation level too strict?
2. Plugin missing required metadata?
3. Plugin has security issues?

**Solution:**
```bash
# Try comprehensive validation for details
python main.py --comprehensive-validate-plugin my_plugin

# Or lower validation level (development only!)
# config.yaml: validation_level: "permissive"
```

### Issue: Name Conflicts

**Symptoms:** Warning about transform name conflicts

**Check:**
1. Plugin transform name same as built-in?
2. Multiple plugins with same transform name?

**Solution:**
```yaml
plugins:
  allow_override_builtin: true  # Allow override (use carefully!)
  # OR rename plugin transform
```

### Issue: Performance Slow

**Symptoms:** Slow plugin loading

**Check:**
1. Too many plugins?
2. Validation level too high?
3. Security scanning enabled?

**Solution:**
```yaml
plugins:
  max_plugins: 20  # Reduce if needed
  validation_level: "standard"  # Lower from "strict"
  security_scanning: false  # Disable for speed
  cache_validation_results: true  # Enable caching
  trusted_plugins:  # Add known-good plugins
    - my_trusted_plugin
```

---

## 📊 Default Values

| Setting | Default | Reason |
|---------|---------|--------|
| `enabled` | `false` | Safe default, no surprise behavior |
| `plugin_paths` | `["./plugins"]` | Standard location |
| `auto_discover` | `true` | Convenient |
| `auto_validate` | `true` | Safety |
| `validation_level` | `"standard"` | Balance security/performance |
| `trusted_plugins` | `[]` | Explicit opt-in |
| `disabled_plugins` | `[]` | Explicit opt-out |
| `allow_experimental` | `false` | Stability |
| `max_plugins` | `50` | Reasonable limit |
| `require_metadata` | `true` | Documentation |
| `enforce_checksums` | `false` | Optional security |
| `security_scanning` | `false` | Optional security |
| `fail_on_plugin_error` | `false` | Robustness |
| `cache_validation_results` | `true` | Performance |
| `allow_override_builtin` | `false` | Safety |
| `warn_on_name_conflict` | `true` | Awareness |

---

## 💡 Pro Tips

### Tip 1: Use Trusted Plugins List

For plugins you use regularly:

```yaml
trusted_plugins:
  - my_daily_transform
  - common_utility
```

**Benefit:** Skip validation, faster loading

### Tip 2: Separate Dev and Prod Configs

**config.dev.yaml:**
```yaml
plugins:
  validation_level: "permissive"
  allow_experimental: true
```

**config.prod.yaml:**
```yaml
plugins:
  validation_level: "strict"
  allow_experimental: false
```

**Usage:**
```bash
python main.py --config config.dev.yaml   # Development
python main.py --config config.prod.yaml  # Production
```

### Tip 3: Version Your Plugin Configs

Track plugin configuration changes in version control:

```bash
git add config.yaml
git commit -m "feat: enable quantum_transforms plugin"
```

### Tip 4: Document Your Plugin Setup

Add comments to your config:

```yaml
plugins:
  enabled: true
  plugin_paths:
    - ./plugins
  # Using custom energy transforms for project XYZ
  # Last validated: 2025-10-13
  # Contact: researcher@university.edu
```

---

## 📚 Further Reading

- **Full Implementation Summary:** `Phase3_Step3.2.5_Implementation_Summary.md`
- **Plugin System Core:** Phase 3.2.1 documentation
- **Plugin Schema:** Phase 3.2.3 documentation  
- **CLI Integration:** Phase 3.2.4 documentation
- **Plugin Distribution:** Phase 3.2.6 (upcoming)
- **Main Pipeline Integration:** Phase 3.2.7 (upcoming)

---

## 🆘 Getting Help

### Documentation

1. Check inline comments in `config.yaml`
2. Read implementation summaries
3. Review plugin.yaml in example plugins

### Debugging

```bash
# Verbose logging
export LOG_LEVEL=DEBUG
python main.py

# List all plugins
python main.py --list-plugins

# Validate specific plugin
python main.py --validate-plugin NAME

# Show plugin details
python main.py --plugin-info NAME
```

### Community

- GitHub Issues: Report bugs or request features
- Discussions: Ask questions, share plugins
- Documentation: Check latest docs

---

**Quick Reference Version:** 1.0.0  
**Last Updated:** October 13, 2025  
**Status:** Ready for use

---

**End of Quick Reference**
