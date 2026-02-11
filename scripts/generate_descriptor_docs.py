"""
Generate comprehensive markdown documentation for VQM24 molecular descriptors.

This script creates a reference document containing:
- Complete list of available molecular descriptors
- Descriptors organized by category
- Table of contents with descriptor counts
- YAML configuration examples

Usage:
    python generate_descriptor_docs.py
    python generate_descriptor_docs.py --output /path/to/output.md
    python generate_descriptor_docs.py --verbose

Author: VQM24 Pipeline Team
License: MIT
"""

import sys
import argparse
from pathlib import Path
from typing import List
from datetime import datetime

from vqm24_pipeline.descriptors.descriptor_registry import DescriptorRegistry
from vqm24_pipeline.descriptors.descriptor_categories import DescriptorCategory


def generate_header(total_descriptors: int) -> List[str]:
    """
    Generate documentation header section.
    
    Args:
        total_descriptors: Total number of descriptors across all categories
        
    Returns:
        List of header markdown lines
    """
    return [
        '# VQM24 Molecular Descriptors Reference\n\n',
        'Complete list of available molecular descriptors organized by category.\n\n',
        f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n',
        f'**Total Descriptors:** {total_descriptors}\n\n',
        '---\n\n'
    ]


def generate_table_of_contents(registry: DescriptorRegistry) -> List[str]:
    """
    Generate table of contents with category links and counts.
    
    Args:
        registry: Initialized DescriptorRegistry instance
        
    Returns:
        List of TOC markdown lines
    """
    output = ['## Table of Contents\n\n']
    
    for cat in DescriptorCategory:
        count = len(registry.list_available_descriptors(category=cat))
        anchor = cat.value.replace("_", "-")
        output.append(f'- [{cat.value.title()}](#{anchor}) ({count} descriptors)\n')
    
    output.append('\n')
    return output


def generate_category_sections(registry: DescriptorRegistry, verbose: bool = False) -> List[str]:
    """
    Generate descriptor sections organized by category.
    
    Args:
        registry: Initialized DescriptorRegistry instance
        verbose: Enable verbose logging for warnings
        
    Returns:
        List of category section markdown lines
    """
    output = []
    
    for cat in DescriptorCategory:
        descs = sorted(registry.list_available_descriptors(category=cat))
        
        # Category header
        output.append(f'## {cat.value.title()}\n\n')
        output.append(f'**Count:** {len(descs)} descriptors\n\n')
        
        # Descriptor list or empty message
        if descs:
            output.append('**Descriptors:**\n\n')
            for desc in descs:
                output.append(f'- `{desc}`\n')
        else:
            output.append('*No descriptors available in this category.*\n')
            if verbose:
                print(f'⚠ Warning: Category "{cat.value}" has no descriptors')
        
        output.append('\n')
    
    return output


def generate_config_example(registry: DescriptorRegistry) -> List[str]:
    """
    Generate YAML configuration example section.
    
    Args:
        registry: Initialized DescriptorRegistry instance
        
    Returns:
        List of configuration example markdown lines
    """
    output = [
        '## Configuration Example\n\n',
        '```yaml\n',
        'descriptors:\n',
        '  enabled: true\n',
        '  default_categories:\n'
    ]
    
    # Default categories list
    for cat in DescriptorCategory:
        output.append(f'    - {cat.value}\n')
    
    output.append('\n  categories:\n')
    
    # Per-category configuration
    for cat in DescriptorCategory:
        descs = registry.list_available_descriptors(category=cat)
        output.append(f'    {cat.value}:\n')
        output.append(f'      enabled: true\n')
        output.append(f'      descriptors: null  # null = all {len(descs)} descriptors\n')
        
        if descs:
            ex1 = descs[0]
            ex2 = descs[1] if len(descs) > 1 else descs[0]
            output.append(f'      # Or specify: ["{ex1}", "{ex2}"]\n')
        
        output.append('\n')
    
    output.append('```\n')
    return output


def generate_documentation(registry: DescriptorRegistry, verbose: bool = False) -> List[str]:
    """
    Generate complete markdown documentation content from descriptor registry.
    
    Args:
        registry: Initialized DescriptorRegistry instance
        verbose: Enable verbose logging
        
    Returns:
        List of strings representing markdown documentation lines
    """
    output = []
    
    total_descriptors = len(registry.list_available_descriptors())
    
    if verbose:
        print(f'ℹ Generating documentation for {total_descriptors} descriptors...')
    
    # Build documentation sections
    output.extend(generate_header(total_descriptors))
    output.extend(generate_table_of_contents(registry))
    output.extend(generate_category_sections(registry, verbose))
    output.extend(generate_config_example(registry))
    
    return output


def write_documentation(content: List[str], output_path: Path, verbose: bool = False) -> None:
    """
    Write documentation content to file.
    
    Args:
        content: List of markdown lines to write
        output_path: Path object for output file
        verbose: Enable verbose logging
        
    Raises:
        IOError: If file cannot be written
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f'ℹ Writing to {output_path}...')
        
        # Write content
        output_path.write_text(''.join(content), encoding='utf-8')
        
    except IOError as e:
        raise IOError(f'Failed to write documentation file: {e}')


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generate VQM24 molecular descriptor documentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_descriptor_docs.py
  python generate_descriptor_docs.py --output ./docs/descriptors.md
  python generate_descriptor_docs.py --verbose
        """
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='/app/vqm24/docs/DESCRIPTOR_REFERENCE.md',
        help='Output file path (default: /app/vqm24/docs/DESCRIPTOR_REFERENCE.md)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize registry
        if args.verbose:
            print('ℹ Initializing DescriptorRegistry...')
        
        registry = DescriptorRegistry.get_instance()
        
        # Generate documentation
        content = generate_documentation(registry, args.verbose)
        
        # Write to file
        output_path = Path(args.output)
        write_documentation(content, output_path, args.verbose)
        
        # Success message
        total_descriptors = len(registry.list_available_descriptors())
        print(f'✓ Created {output_path}')
        print(f'✓ Total descriptors documented: {total_descriptors}')
        
        return 0
        
    except Exception as e:
        print(f'✗ Error: {e}', file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
