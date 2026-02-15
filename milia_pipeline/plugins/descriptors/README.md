# VQM24 Descriptor Plugins

This directory contains descriptor plugins for the VQM24 Pipeline.

## Overview

Descriptor plugins allow you to extend VQM24 with custom molecular descriptors without modifying the core codebase.

## Directory Structure

- `example_descriptors/` - Working example plugin demonstrating best practices
- `user_template/` - Template for creating your own plugins
- Your plugins should be added as subdirectories here

## Quick Start

### Creating a Plugin

1. Copy the `user_template/` directory:
```bash
   cp -r user_template/ my_descriptors/
```

2. Edit `my_descriptors/plugin.yaml`:
   - Set your plugin name, version, author
   - Declare your descriptors

3. Implement descriptors in `my_descriptors/descriptors.py`:
```python
   def my_descriptor(mol):
       '''Calculate my custom descriptor'''
       # Your implementation here
       return value
```

4. Test your plugin:
```python
   from vqm24_pipeline.descriptors.descriptor_plugin_system import discover_plugins
   discover_plugins(paths=[Path("./vqm24_pipeline/plugins/descriptors")])
```

## Plugin Structure

### Required Files

1. **plugin.yaml** - Plugin metadata and descriptor declarations
2. **descriptors.py** - Descriptor function implementations

### plugin.yaml Format
```yaml
plugin_name: "my_descriptors"
version: "1.0.0"
author: "Your Name"
email: "your.email@example.com"
description: "Brief description of your plugin"
license: "MIT"
vqm24_version: ">=1.0.0"
python_version: ">=3.8"

descriptors:
  - name: "MyDescriptor"
    function_name: "calculate_my_descriptor"
    module_path: "descriptors"
    category: "constitutional"  # constitutional, topological, electronic, geometric, drug_likeness, fragments
    description: "Description of what this descriptor calculates"
    requires_3d: false
    requires_charges: false
```

### Descriptor Function Requirements

Descriptor functions must:
1. Take an RDKit Mol object as the first parameter
2. Return a numeric value (float or int)
3. Handle errors gracefully (return NaN on failure)

Example:
```python
import math
from rdkit import Chem

def calculate_my_descriptor(mol):
    '''
    Calculate my custom descriptor.

    Args:
        mol: RDKit Mol object

    Returns:
        float: Descriptor value
    '''
    try:
        # Your calculation here
        value = 0.0

        # Example: count aromatic atoms
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic():
                value += 1.0

        return value
    except Exception as e:
        # Return NaN on error
        return float('nan')
```

## Best Practices

1. **Error Handling**: Always wrap calculations in try-except
2. **Documentation**: Add docstrings to all functions
3. **Testing**: Test with various molecule types
4. **Performance**: Optimize for batch processing
5. **Validation**: Use `descriptor_plugin_system.validate_plugin()`

## Examples

See `example_descriptors/` for a complete working example.

## Troubleshooting

### Plugin Not Discovered
- Check plugin.yaml syntax (use YAML validator)
- Ensure plugin directory is in the correct location
- Check logs for error messages

### Descriptor Not Working
- Verify function name matches declaration
- Check function signature (must accept mol parameter)
- Test function independently before registering

### Import Errors
- Check all dependencies are installed
- Verify Python version compatibility
- Check for circular imports

## Support

For issues or questions:
1. Check the documentation
2. Review example plugins
3. Check logs for detailed error messages
4. Open an issue on GitHub
