#!/usr/bin/env python3
"""
Analyze Python imports in a project and create a reverse dependency map.
Shows for each module which other modules import it.
Excludes test directories by default.
"""

import ast
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def should_exclude_path(path: Path, exclude_patterns: List[str]) -> bool:
    """Check if a path should be excluded based on patterns."""
    path_str = str(path)
    path_parts = path.parts
    
    for pattern in exclude_patterns:
        # Check if pattern is in any part of the path
        if pattern in path_parts:
            return True
        # Also check the string representation
        if pattern in path_str:
            return True
    
    return False


def get_python_files(root_dir: str, exclude_patterns: List[str] = None) -> List[Path]:
    """Recursively find all Python files in the directory, excluding specified patterns."""
    if exclude_patterns is None:
        exclude_patterns = ['__pycache__', '.venv', '.git', 'tests', 'test', '.pytest_cache', 'dist', 'build', '.tox']
    
    python_files = []
    root_path = Path(root_dir)
    
    for path in root_path.rglob("*.py"):
        if not should_exclude_path(path, exclude_patterns):
            python_files.append(path)
    
    return sorted(python_files)


def get_module_name(file_path: Path, root_dir: Path) -> str:
    """Convert file path to module name."""
    try:
        relative_path = file_path.relative_to(root_dir)
    except ValueError:
        return str(file_path)
    
    # Convert path to module name
    parts = list(relative_path.parts)
    
    # Remove .py extension
    if parts[-1].endswith('.py'):
        parts[-1] = parts[-1][:-3]
    
    # Remove __init__ from the end
    if parts[-1] == '__init__':
        parts = parts[:-1]
    
    return '.'.join(parts) if parts else str(file_path)


def extract_imports(file_path: Path) -> List[Tuple[str, int]]:
    """
    Extract all imports from a Python file with their full module paths.
    Returns list of (module_path, line_number) tuples.
    """
    imports = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Keep the full module path
                    imports.append((alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Keep the full module path
                    imports.append((node.module, node.lineno))
                elif node.level > 0:
                    # Relative import - we'll skip these for now
                    pass
    
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
    
    return imports


def filter_project_imports(imports: List[Tuple[str, int]], all_module_names: Set[str]) -> Set[str]:
    """
    Filter imports to only include modules that exist in the project.
    """
    project_imports = set()
    
    for import_path, _ in imports:
        # Direct match - the import exactly matches a module
        if import_path in all_module_names:
            project_imports.add(import_path)
            continue
        
        # Check if the import is a parent package of any module
        # e.g., import "vqm24_pipeline.config" when we have "vqm24_pipeline.config.config_loader"
        for module_name in all_module_names:
            if module_name.startswith(import_path + '.'):
                # Don't add the package, add the actual module
                # Actually, we should add the import as-is if it's a package import
                project_imports.add(import_path)
                break
        
        # Check if the import is a submodule of a known module
        parts = import_path.split('.')
        for i in range(len(parts), 0, -1):
            partial = '.'.join(parts[:i])
            if partial in all_module_names:
                project_imports.add(partial)
                break
    
    return project_imports


def create_reverse_dependency_map(module_imports: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Create a reverse map showing which modules import each module."""
    reverse_map = defaultdict(list)
    
    # Initialize all modules
    for module in module_imports:
        if module not in reverse_map:
            reverse_map[module] = []
    
    # Build reverse dependencies
    for importing_module, imported_modules in module_imports.items():
        for imported_module in imported_modules:
            reverse_map[imported_module].append(importing_module)
    
    # Sort the lists for consistent output and remove duplicates
    for module in reverse_map:
        reverse_map[module] = sorted(set(reverse_map[module]))
    
    return dict(sorted(reverse_map.items()))


def main():
    project_root = Path("/app/vqm24")
    
    if not project_root.exists():
        print(f"Error: Directory {project_root} does not exist!")
        print("Please edit line 131 of this script to set the correct project path.")
        return
    
    print(f"Analyzing Python project at: {project_root}")
    print("=" * 80)
    
    # Find all Python files (excluding tests and other patterns)
    exclude_patterns = ['__pycache__', '.venv', '.git', 'tests', 'test', '.pytest_cache', 'dist', 'build', '.tox']
    python_files = get_python_files(project_root, exclude_patterns=exclude_patterns)
    
    print(f"\nFound {len(python_files)} Python files")
    print(f"Excluded patterns: {', '.join(exclude_patterns)}")
    
    # First pass: get all module names
    all_module_names = set()
    file_to_module = {}
    
    print("\n" + "=" * 80)
    print("MODULE MAPPING")
    print("=" * 80)
    for file_path in python_files:
        module_name = get_module_name(file_path, project_root)
        all_module_names.add(module_name)
        file_to_module[file_path] = module_name
        print(f"{module_name}")
    
    print(f"\nTotal modules: {len(all_module_names)}")
    
    # Second pass: extract imports from each file
    module_imports = {}
    
    print("\n" + "=" * 80)
    print("EXTRACTING IMPORTS")
    print("=" * 80)
    
    for file_path in python_files:
        module_name = file_to_module[file_path]
        imports = extract_imports(file_path)
        
        # Filter to only project-internal imports
        project_imports = filter_project_imports(imports, all_module_names)
        module_imports[module_name] = project_imports
        
        if project_imports:
            print(f"\n{module_name} imports:")
            for imp in sorted(project_imports):
                print(f"  - {imp}")
    
    # Create reverse dependency map
    print("\n" + "=" * 80)
    print("REVERSE DEPENDENCY MAP")
    print("(For each module, shows which OTHER PROJECT MODULES import it)")
    print("=" * 80)
    
    reverse_map = create_reverse_dependency_map(module_imports)
    
    for module in sorted(reverse_map.keys()):
        importers = reverse_map[module]
        if importers:
            print(f"\n{module}:")
            print(f"  Used by {len(importers)} module(s):")
            for importer in importers:
                print(f"    - {importer}")
    
    # Show modules with no importers
    print("\n" + "=" * 80)
    print("MODULES NOT IMPORTED BY ANY OTHER MODULE")
    print("=" * 80)
    unused_modules = [m for m, users in reverse_map.items() if not users]
    for module in sorted(unused_modules):
        print(f"  - {module}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total modules analyzed: {len(module_imports)}")
    
    used_modules = [m for m, users in reverse_map.items() if users]
    
    print(f"Modules imported by others: {len(used_modules)}")
    print(f"Modules not imported by others: {len(unused_modules)}")
    
    # Most depended upon modules
    most_used = sorted(reverse_map.items(), key=lambda x: len(x[1]), reverse=True)[:20]
    print(f"\nTop 20 most imported modules:")
    for module, importers in most_used:
        if importers:
            print(f"  {module}: {len(importers)} importers")
    
    # Save to file
    output_file = project_root / "reverse_dependencies.txt"
    try:
        with open(output_file, 'w') as f:
            f.write("REVERSE DEPENDENCY MAP\n")
            f.write("(For each module, shows which OTHER PROJECT MODULES import it)\n")
            f.write("=" * 80 + "\n\n")
            
            for module in sorted(reverse_map.keys()):
                importers = reverse_map[module]
                f.write(f"{module}:\n")
                if importers:
                    f.write(f"  Used by {len(importers)} module(s):\n")
                    for importer in importers:
                        f.write(f"    - {importer}\n")
                else:
                    f.write(f"  Not imported by any other module\n")
                f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total modules analyzed: {len(module_imports)}\n")
            f.write(f"Modules imported by others: {len(used_modules)}\n")
            f.write(f"Modules not imported by others: {len(unused_modules)}\n")
        
        print(f"\nDetailed report saved to: {output_file}")
    except Exception as e:
        print(f"\nWarning: Could not save report to {output_file}: {e}")
        print("Report printed to console only.")


if __name__ == "__main__":
    main()
