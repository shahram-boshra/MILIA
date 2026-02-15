#!/usr/bin/env python3
"""
Integration tests for descriptor plugin system.

Tests the complete flow from plugin discovery to descriptor calculation.
This version is adapted for Docker environment with proper path handling.

Production-Ready Integration Test Suite:
- End-to-end plugin discovery and registration workflows
- Cross-component interaction testing (plugin_loader ↔ registry)
- Error recovery and graceful degradation testing
- Plugin lifecycle management (enable/disable)
- Concurrent access and thread safety
- Real-world plugin scenarios with actual RDKit calculations
"""

import sys
import threading
import time
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import pytest
from rdkit import Chem

from milia_pipeline.descriptors.descriptor_plugin_system import (
    DescriptorPluginLoader,
    discover_plugins,
    get_plugin_info,
    list_plugins,
    plugin_loader,
    validate_plugin,
)
from milia_pipeline.descriptors.descriptor_registry import (
    get_descriptor,
    has_descriptor,
    registry,
)


def clear_plugin_loader():
    """
    Properly clear singleton plugin loader - prevents state bleeding between tests.

    The DescriptorPluginLoader uses _instance (singular), not _instances (dict).
    We reset the existing singleton instance rather than destroying it to avoid lock conflicts.
    """
    # Get the existing singleton instance
    existing_loader = DescriptorPluginLoader._instance

    if existing_loader is not None:
        # Clear all internal state
        if hasattr(existing_loader, "_plugin_paths"):
            existing_loader._plugin_paths.clear()
        if hasattr(existing_loader, "_plugins"):
            existing_loader._plugins.clear()
        if hasattr(existing_loader, "_enabled_plugins"):
            existing_loader._enabled_plugins.clear()
        if hasattr(existing_loader, "_disabled_plugins"):
            existing_loader._disabled_plugins.clear()

        # Clear the _initialized flag so __init__ can run again if needed
        if hasattr(existing_loader, "_initialized"):
            delattr(existing_loader, "_initialized")
            # Re-initialize to get fresh data structures
            existing_loader.__init__()


def clear_registry():
    """
    Properly clear singleton registry.

    Don't clear _instances. Just reset the existing singleton's data to avoid lock conflicts.
    """
    # Get the existing singleton instance (don't create new one)
    existing_registry = registry.__class__._instances.get(registry.__class__)

    if existing_registry is not None:
        # Simply reset the existing instance's data
        # This preserves the singleton and avoids lock conflicts
        existing_registry.reset()

        # Also clear the _initialized flag so __init__ can run again if needed
        if hasattr(existing_registry, "_initialized"):
            delattr(existing_registry, "_initialized")
            # Re-initialize to get fresh data structures
            existing_registry.__init__()


class TestDescriptorPluginIntegration:
    """Integration tests for complete plugin workflow"""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset registry and loader before and after each test"""
        clear_plugin_loader()
        clear_registry()
        yield
        clear_plugin_loader()
        clear_registry()

    def test_example_plugin_discovery(self):
        """Test discovery of example plugin"""
        # Use absolute path from project root
        plugin_dir = project_root / "milia_pipeline" / "plugins" / "descriptors"

        if not plugin_dir.exists():
            pytest.skip(f"Plugin directory not found at {plugin_dir}")

        plugins = discover_plugins(paths=[plugin_dir])
        assert "example_descriptors" in plugins, f"example_descriptors not found in {plugins}"

    def test_example_plugin_descriptors(self):
        """Test that example plugin descriptors work"""
        # Use absolute path from project root
        plugin_dir = project_root / "milia_pipeline" / "plugins" / "descriptors"

        if not plugin_dir.exists():
            pytest.skip(f"Plugin directory not found at {plugin_dir}")

        discover_plugins(paths=[plugin_dir])

        # Check descriptors are registered
        assert registry.has_descriptor("AromaticRatio"), "AromaticRatio descriptor not registered"
        assert registry.has_descriptor("HeteroatomRatio"), (
            "HeteroatomRatio descriptor not registered"
        )
        assert registry.has_descriptor("ChainLength"), "ChainLength descriptor not registered"

        # Test descriptors
        mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene

        calc_aromatic = registry.get_descriptor("AromaticRatio")
        value = calc_aromatic(mol)
        assert value == 1.0, f"Expected AromaticRatio=1.0 for benzene, got {value}"

        calc_hetero = registry.get_descriptor("HeteroatomRatio")
        value = calc_hetero(mol)
        assert value == 0.0, f"Expected HeteroatomRatio=0.0 for benzene, got {value}"

    def test_plugin_validation(self):
        """Test plugin validation"""
        # Use absolute path from project root
        plugin_dir = project_root / "milia_pipeline" / "plugins" / "descriptors"

        if not plugin_dir.exists():
            pytest.skip(f"Plugin directory not found at {plugin_dir}")

        discover_plugins(paths=[plugin_dir])

        is_valid, errors = validate_plugin("example_descriptors")
        assert is_valid, f"Plugin validation failed with errors: {errors}"
        assert len(errors) == 0, f"Unexpected validation errors: {errors}"

    def test_bonus_descriptor_discovery(self):
        """Test that undeclared descriptors are discovered"""
        # Use absolute path from project root
        plugin_dir = project_root / "milia_pipeline" / "plugins" / "descriptors"

        if not plugin_dir.exists():
            pytest.skip(f"Plugin directory not found at {plugin_dir}")

        discover_plugins(paths=[plugin_dir])

        # ring_complexity is not in plugin.yaml but should be discovered
        if registry.has_descriptor("ring_complexity"):
            calc_ring = registry.get_descriptor("ring_complexity")
            mol = Chem.MolFromSmiles("C1CCCCC1")  # Cyclohexane
            value = calc_ring(mol)
            assert value > 0, f"Expected positive ring_complexity for cyclohexane, got {value}"

    def test_multiple_plugins(self, tmp_path):
        """Test loading multiple plugins"""
        # Create two test plugins
        for i in range(2):
            plugin_dir = tmp_path / f"plugin_{i}"
            plugin_dir.mkdir()

            (plugin_dir / "plugin.yaml").write_text(f"""
plugin_name: "test_plugin_{i}"
version: "1.0.0"
author: "Test"
description: "Test plugin {i}"
milia_version: ">=1.0.0"

descriptors:
  - name: "TestDesc{i}"
    function_name: "calc_test{i}"
    module_path: "descriptors"
    category: "constitutional"
    description: "Test descriptor {i}"
""")

            (plugin_dir / "descriptors.py").write_text(f"""
def calc_test{i}(mol):
    '''Test descriptor function {i}'''
    return {i}.0
""")

        plugins = discover_plugins(paths=[tmp_path])
        assert len(plugins) == 2, f"Expected 2 plugins, found {len(plugins)}"
        assert "test_plugin_0" in plugins, f"test_plugin_0 not found in {plugins}"
        assert "test_plugin_1" in plugins, f"test_plugin_1 not found in {plugins}"

    def test_end_to_end_workflow(self):
        """Test complete workflow from discovery to calculation"""
        # Use absolute path from project root
        plugin_dir = project_root / "milia_pipeline" / "plugins" / "descriptors"

        if not plugin_dir.exists():
            pytest.skip(f"Plugin directory not found at {plugin_dir}")

        # 1. Auto-discover RDKit descriptors
        registry.auto_discover_rdkit_descriptors()
        builtin_count = len(registry.list_available_descriptors())
        assert builtin_count > 0, "No built-in RDKit descriptors discovered"

        # 2. Discover plugins
        plugins = discover_plugins(paths=[plugin_dir])
        assert len(plugins) > 0, f"No plugins discovered in {plugin_dir}"

        # 3. Validate plugins
        for plugin in plugins:
            is_valid, errors = validate_plugin(plugin)
            assert is_valid, f"Plugin {plugin} validation failed: {errors}"

        # 4. Check total descriptor count increased
        total_count = len(registry.list_available_descriptors())
        assert total_count > builtin_count, (
            f"Total descriptor count ({total_count}) should be greater than "
            f"builtin count ({builtin_count}) after plugin discovery"
        )

        # 5. Calculate descriptors
        mol = Chem.MolFromSmiles("CCO")  # Ethanol
        assert mol is not None, "Failed to create ethanol molecule"

        # Built-in descriptor
        if registry.has_descriptor("MolWt"):
            molwt_func = registry.get_descriptor("MolWt")
            molwt = molwt_func(mol)
            assert molwt > 0, f"Expected positive MolWt for ethanol, got {molwt}"
            # Ethanol MW should be around 46 g/mol
            assert 45 < molwt < 47, f"Expected MolWt ~46 for ethanol, got {molwt}"

        # Plugin descriptor
        if registry.has_descriptor("AromaticRatio"):
            aromatic_func = registry.get_descriptor("AromaticRatio")
            aromatic_ratio = aromatic_func(mol)
            assert aromatic_ratio == 0.0, (
                f"Expected AromaticRatio=0.0 for ethanol (no aromatic atoms), got {aromatic_ratio}"
            )

    def test_plugin_descriptor_metadata(self):
        """Test that plugin descriptors have proper metadata"""
        # Use absolute path from project root
        plugin_dir = project_root / "milia_pipeline" / "plugins" / "descriptors"

        if not plugin_dir.exists():
            pytest.skip(f"Plugin directory not found at {plugin_dir}")

        discover_plugins(paths=[plugin_dir])

        # Check that plugin descriptors are listed
        all_descriptors = registry.list_available_descriptors()
        assert len(all_descriptors) > 0, "No descriptors available after plugin discovery"

        # Check that plugin descriptors can be retrieved and called
        if registry.has_descriptor("AromaticRatio"):
            func = registry.get_descriptor("AromaticRatio")
            assert func is not None, "AromaticRatio function is None"
            assert callable(func), "AromaticRatio should be callable"

            # Test that it works
            mol = Chem.MolFromSmiles("c1ccccc1")
            value = func(mol)
            assert value == 1.0, f"Expected AromaticRatio=1.0 for benzene, got {value}"

    def test_plugin_enable_disable(self):
        """Test enabling and disabling plugins"""
        # Use absolute path from project root
        plugin_dir = project_root / "milia_pipeline" / "plugins" / "descriptors"

        if not plugin_dir.exists():
            pytest.skip(f"Plugin directory not found at {plugin_dir}")

        plugins = discover_plugins(paths=[plugin_dir])

        if "example_descriptors" not in plugins:
            pytest.skip("example_descriptors plugin not found")

        # Plugins are NOT enabled by default after discovery, so explicitly enable it
        plugin_loader.enable_plugin("example_descriptors")
        assert plugin_loader.is_enabled("example_descriptors"), (
            "Plugin should be enabled after enable_plugin call"
        )

        # Disable the plugin
        plugin_loader.disable_plugin("example_descriptors")
        assert not plugin_loader.is_enabled("example_descriptors"), "Plugin should be disabled"

        # Re-enable the plugin
        plugin_loader.enable_plugin("example_descriptors")
        assert plugin_loader.is_enabled("example_descriptors"), "Plugin should be re-enabled"

    def test_plugin_info_retrieval(self):
        """Test retrieving plugin information"""
        # Use absolute path from project root
        plugin_dir = project_root / "milia_pipeline" / "plugins" / "descriptors"

        if not plugin_dir.exists():
            pytest.skip(f"Plugin directory not found at {plugin_dir}")

        plugins = discover_plugins(paths=[plugin_dir])

        if "example_descriptors" not in plugins:
            pytest.skip("example_descriptors plugin not found")

        # Get plugin info
        info = plugin_loader.get_plugin_info("example_descriptors")
        assert info is not None, "Plugin info should not be None"
        assert info["plugin_name"] == "example_descriptors", (
            f"Expected plugin_name='example_descriptors', got {info['plugin_name']}"
        )
        assert "version" in info, "Plugin info should contain version"
        assert "author" in info, "Plugin info should contain author"
        assert "descriptor_declarations" in info, (
            "Plugin info should contain descriptor_declarations"
        )
        assert "registered_descriptors" in info, "Plugin info should contain registered_descriptors"


class TestPluginErrorHandling:
    """Test error handling in plugin system"""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset registry and loader before and after each test"""
        clear_plugin_loader()
        clear_registry()
        yield
        clear_plugin_loader()
        clear_registry()

    def test_invalid_yaml_handling(self, tmp_path):
        """Test handling of invalid YAML files"""
        plugin_dir = tmp_path / "invalid_yaml_plugin"
        plugin_dir.mkdir()

        # Create invalid YAML
        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "invalid_yaml"
version: 1.0.0  # Missing quotes
author: Test
this is: invalid: yaml: [
""")

        # Should not crash, just skip the invalid plugin
        plugins = discover_plugins(paths=[tmp_path])
        # The plugin may or may not be discovered depending on YAML parser
        # The important thing is it doesn't crash
        assert isinstance(plugins, list), "discover_plugins should return a list"

    def test_missing_descriptor_module(self, tmp_path):
        """Test handling of missing descriptor module"""
        plugin_dir = tmp_path / "missing_module_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "missing_module_plugin"
version: "1.0.0"
author: "Test"
description: "Plugin with missing module"
milia_version: ">=1.0.0"

descriptors:
  - name: "MissingDesc"
    function_name: "calc_missing"
    module_path: "nonexistent_module"
    category: "constitutional"
    description: "Descriptor with missing module"
""")

        # Should handle gracefully
        plugins = discover_plugins(paths=[tmp_path])

        if "missing_module_plugin" in plugins:
            # If plugin was discovered, validation should fail
            is_valid, errors = validate_plugin("missing_module_plugin")
            assert not is_valid or len(errors) > 0, "Validation should fail for missing module"

    def test_missing_descriptor_function(self, tmp_path):
        """Test handling of missing descriptor function"""
        plugin_dir = tmp_path / "missing_function_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "missing_function_plugin"
version: "1.0.0"
author: "Test"
description: "Plugin with missing function"
milia_version: ">=1.0.0"

descriptors:
  - name: "MissingFuncDesc"
    function_name: "nonexistent_function"
    module_path: "descriptors"
    category: "constitutional"
    description: "Descriptor with missing function"
""")

        (plugin_dir / "descriptors.py").write_text("""
def some_other_function(mol):
    return 0.0
""")

        # Should handle gracefully
        plugins = discover_plugins(paths=[tmp_path])

        if "missing_function_plugin" in plugins:
            # If plugin was discovered, validation should fail
            is_valid, errors = validate_plugin("missing_function_plugin")
            assert not is_valid or len(errors) > 0, "Validation should fail for missing function"


class TestPluginIsolation:
    """Test that plugins are properly isolated"""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset registry and loader before and after each test"""
        clear_plugin_loader()
        clear_registry()
        yield
        clear_plugin_loader()
        clear_registry()

    def test_plugin_descriptor_name_collision(self, tmp_path):
        """Test handling of descriptor name collisions between plugins"""
        # Create two plugins with same descriptor name
        for i in range(2):
            plugin_dir = tmp_path / f"collision_plugin_{i}"
            plugin_dir.mkdir()

            (plugin_dir / "plugin.yaml").write_text(f"""
plugin_name: "collision_plugin_{i}"
version: "1.0.0"
author: "Test"
description: "Collision test plugin {i}"
milia_version: ">=1.0.0"

descriptors:
  - name: "CollisionDesc"
    function_name: "calc_collision"
    module_path: "descriptors"
    category: "constitutional"
    description: "Colliding descriptor"
""")

            (plugin_dir / "descriptors.py").write_text(f"""
def calc_collision(mol):
    return {i}.0
""")

        # Discover both plugins
        plugins = discover_plugins(paths=[tmp_path])

        # Should handle the collision (e.g., last one wins, or warning)
        assert len(plugins) == 2, f"Expected 2 plugins, found {len(plugins)}"

        # Check which value is actually registered
        if registry.has_descriptor("CollisionDesc"):
            func = registry.get_descriptor("CollisionDesc")
            mol = Chem.MolFromSmiles("C")
            value = func(mol)
            # Value should be either 0.0 or 1.0 depending on which plugin won
            assert value in [0.0, 1.0], (
                f"Expected collision resolution to give 0.0 or 1.0, got {value}"
            )


class TestRegistryPluginIntegration:
    """Test integration between plugin system and descriptor registry"""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset registry and loader before and after each test"""
        clear_plugin_loader()
        clear_registry()
        yield
        clear_plugin_loader()
        clear_registry()

    def test_plugin_descriptors_appear_in_registry(self, tmp_path):
        """Test that plugin descriptors are properly registered in the global registry"""
        plugin_dir = tmp_path / "registry_test_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "registry_test_plugin"
version: "1.0.0"
author: "Test"
description: "Registry integration test"
milia_version: ">=1.0.0"

descriptors:
  - name: "RegistryTestDesc"
    function_name: "calc_registry_test"
    module_path: "descriptors"
    category: "constitutional"
    description: "Test descriptor for registry integration"
""")

        (plugin_dir / "descriptors.py").write_text("""
def calc_registry_test(mol):
    '''Returns the number of atoms'''
    return mol.GetNumAtoms() if mol else 0
""")

        # Before discovery
        assert not registry.has_descriptor("RegistryTestDesc")

        # Discover plugins
        plugins = discover_plugins(paths=[tmp_path])
        assert "registry_test_plugin" in plugins

        # After discovery - descriptor should be in registry
        assert registry.has_descriptor("RegistryTestDesc")

        # Test the descriptor works
        func = registry.get_descriptor("RegistryTestDesc")
        assert func is not None
        assert callable(func)

        mol = Chem.MolFromSmiles("CCO")  # Ethanol - 3 heavy atoms + 6 H = 9
        # Note: GetNumAtoms returns total atoms (with implicit H based on RDKit settings)
        result = func(mol)
        assert result > 0

    def test_plugin_descriptors_in_listing(self, tmp_path):
        """Test that plugin descriptors appear in registry listing"""
        plugin_dir = tmp_path / "listing_test_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "listing_test_plugin"
version: "1.0.0"
author: "Test"

descriptors:
  - name: "ListingTestDesc1"
    function_name: "calc_listing1"
    module_path: "descriptors"
    category: "constitutional"
  - name: "ListingTestDesc2"
    function_name: "calc_listing2"
    module_path: "descriptors"
    category: "constitutional"
""")

        (plugin_dir / "descriptors.py").write_text("""
def calc_listing1(mol):
    return 1.0

def calc_listing2(mol):
    return 2.0
""")

        discover_plugins(paths=[tmp_path])

        all_descriptors = registry.list_available_descriptors()
        assert "ListingTestDesc1" in all_descriptors
        assert "ListingTestDesc2" in all_descriptors

    def test_plugin_descriptors_metadata_in_registry(self, tmp_path):
        """Test that plugin descriptor metadata is properly stored in registry"""
        plugin_dir = tmp_path / "metadata_test_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "metadata_test_plugin"
version: "1.0.0"
author: "Test"

descriptors:
  - name: "MetadataTestDesc"
    function_name: "calc_metadata_test"
    module_path: "descriptors"
    category: "electronic"
    description: "A descriptor with specific metadata"
    requires_3d: false
    requires_charges: false
""")

        (plugin_dir / "descriptors.py").write_text("""
def calc_metadata_test(mol):
    return 42.0
""")

        discover_plugins(paths=[tmp_path])

        # Get metadata from registry
        metadata = registry.get_metadata("MetadataTestDesc")
        assert metadata is not None
        assert metadata.name == "MetadataTestDesc"

    def test_plugin_descriptors_in_statistics(self, tmp_path):
        """Test that plugin descriptors are counted in registry statistics"""
        plugin_dir = tmp_path / "stats_test_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "stats_test_plugin"
version: "1.0.0"
author: "Test"

descriptors:
  - name: "StatsTestDesc"
    function_name: "calc_stats_test"
    module_path: "descriptors"
    category: "constitutional"
""")

        (plugin_dir / "descriptors.py").write_text("""
def calc_stats_test(mol):
    return 1.0
""")

        # Get stats before
        stats_before = registry.get_statistics()
        total_before = stats_before["total_descriptors"]
        plugins_before = stats_before["plugin_descriptors"]

        discover_plugins(paths=[tmp_path])

        # Get stats after
        stats_after = registry.get_statistics()
        total_after = stats_after["total_descriptors"]
        plugins_after = stats_after["plugin_descriptors"]

        # Should have more descriptors
        assert total_after >= total_before
        assert plugins_after >= plugins_before

    def test_get_plugin_descriptors_mapping(self, tmp_path):
        """Test getting the mapping of descriptors to plugin names"""
        plugin_dir = tmp_path / "mapping_test_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "mapping_test_plugin"
version: "1.0.0"
author: "Test"

descriptors:
  - name: "MappingTestDesc"
    function_name: "calc_mapping_test"
    module_path: "descriptors"
    category: "constitutional"
""")

        (plugin_dir / "descriptors.py").write_text("""
def calc_mapping_test(mol):
    return 1.0
""")

        discover_plugins(paths=[tmp_path])

        plugin_desc_map = registry.get_plugin_descriptors()

        if "MappingTestDesc" in plugin_desc_map:
            assert plugin_desc_map["MappingTestDesc"] == "mapping_test_plugin"


class TestConvenienceFunctionsIntegration:
    """Test module-level convenience functions work end-to-end"""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset registry and loader before and after each test"""
        clear_plugin_loader()
        clear_registry()
        yield
        clear_plugin_loader()
        clear_registry()

    def test_list_plugins_after_discovery(self, tmp_path):
        """Test list_plugins convenience function"""
        plugin_dir = tmp_path / "list_conv_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "list_conv_plugin"
version: "1.0.0"
author: "Test"

descriptors: []
""")

        discover_plugins(paths=[tmp_path])

        plugins = list_plugins()
        assert "list_conv_plugin" in plugins

    def test_get_plugin_info_after_discovery(self, tmp_path):
        """Test get_plugin_info convenience function"""
        plugin_dir = tmp_path / "info_conv_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "info_conv_plugin"
version: "2.5.0"
author: "Integration Test Author"
description: "Plugin for testing get_plugin_info"

descriptors: []
""")

        discover_plugins(paths=[tmp_path])

        info = get_plugin_info("info_conv_plugin")
        assert info is not None
        assert info["plugin_name"] == "info_conv_plugin"
        assert info["version"] == "2.5.0"
        assert info["author"] == "Integration Test Author"

    def test_validate_plugin_after_discovery(self, tmp_path):
        """Test validate_plugin convenience function"""
        plugin_dir = tmp_path / "validate_conv_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "validate_conv_plugin"
version: "1.0.0"
author: "Test"

descriptors:
  - name: "ValidateConvDesc"
    function_name: "calc_validate_conv"
    module_path: "descriptors"
    category: "constitutional"
""")

        (plugin_dir / "descriptors.py").write_text("""
def calc_validate_conv(mol):
    return 1.0
""")

        discover_plugins(paths=[tmp_path])

        is_valid, errors = validate_plugin("validate_conv_plugin")
        assert is_valid
        assert len(errors) == 0

    def test_registry_convenience_functions(self, tmp_path):
        """Test registry-level convenience functions"""
        plugin_dir = tmp_path / "reg_conv_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "reg_conv_plugin"
version: "1.0.0"
author: "Test"

descriptors:
  - name: "RegConvDesc"
    function_name: "calc_reg_conv"
    module_path: "descriptors"
    category: "constitutional"
""")

        (plugin_dir / "descriptors.py").write_text("""
def calc_reg_conv(mol):
    return mol.GetNumAtoms() if mol else 0
""")

        discover_plugins(paths=[tmp_path])

        # Test has_descriptor convenience function
        assert has_descriptor("RegConvDesc")

        # Test get_descriptor convenience function
        func = get_descriptor("RegConvDesc")
        assert func is not None

        mol = Chem.MolFromSmiles("C")
        result = func(mol)
        assert result > 0


class TestConcurrentPluginAccess:
    """Test concurrent access to plugin system and registry"""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset registry and loader before and after each test"""
        clear_plugin_loader()
        clear_registry()
        yield
        clear_plugin_loader()
        clear_registry()

    def test_concurrent_descriptor_calculation(self, tmp_path):
        """Test concurrent descriptor calculations from plugin"""
        plugin_dir = tmp_path / "concurrent_calc_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "concurrent_calc_plugin"
version: "1.0.0"
author: "Test"

descriptors:
  - name: "ConcurrentCalcDesc"
    function_name: "calc_concurrent"
    module_path: "descriptors"
    category: "constitutional"
""")

        (plugin_dir / "descriptors.py").write_text("""
def calc_concurrent(mol):
    '''Thread-safe descriptor calculation'''
    if mol is None:
        return 0.0
    return float(mol.GetNumAtoms())
""")

        discover_plugins(paths=[tmp_path])

        # Get the descriptor function
        calc_func = registry.get_descriptor("ConcurrentCalcDesc")
        assert calc_func is not None

        # Test molecules
        smiles_list = ["C", "CC", "CCC", "CCCC", "c1ccccc1", "CCO", "CCCO"]
        results = {}
        errors = []

        def calculate_descriptor(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                result = calc_func(mol)
                results[smiles] = result
            except Exception as e:
                errors.append((smiles, str(e)))

        # Run calculations concurrently
        threads = [threading.Thread(target=calculate_descriptor, args=(s,)) for s in smiles_list]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0, f"Concurrent calculation errors: {errors}"

        # Verify all calculations completed
        assert len(results) == len(smiles_list)

        # Verify results are correct
        for smiles, result in results.items():
            mol = Chem.MolFromSmiles(smiles)
            assert result == float(mol.GetNumAtoms())

    def test_concurrent_plugin_discovery_and_calculation(self, tmp_path):
        """Test that calculations work while discovery might be happening"""
        # Create plugin
        plugin_dir = tmp_path / "concurrent_disc_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "concurrent_disc_plugin"
version: "1.0.0"
author: "Test"

descriptors:
  - name: "ConcurrentDiscDesc"
    function_name: "calc_conc_disc"
    module_path: "descriptors"
    category: "constitutional"
""")

        (plugin_dir / "descriptors.py").write_text("""
def calc_conc_disc(mol):
    return 42.0
""")

        # Discover first
        discover_plugins(paths=[tmp_path])

        errors = []
        results = []

        def run_calculations():
            try:
                for _ in range(10):
                    if registry.has_descriptor("ConcurrentDiscDesc"):
                        func = registry.get_descriptor("ConcurrentDiscDesc")
                        mol = Chem.MolFromSmiles("C")
                        result = func(mol)
                        results.append(result)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(str(e))

        # Run multiple calculation threads
        threads = [threading.Thread(target=run_calculations) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0, f"Concurrent errors: {errors}"
        # Should have some results
        assert len(results) > 0


class TestRealWorldScenarios:
    """Test real-world usage scenarios"""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset registry and loader before and after each test"""
        clear_plugin_loader()
        clear_registry()
        yield
        clear_plugin_loader()
        clear_registry()

    def test_chemical_series_analysis(self, tmp_path):
        """Test analyzing a series of related molecules with plugin descriptors"""
        plugin_dir = tmp_path / "series_analysis_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "series_analysis_plugin"
version: "1.0.0"
author: "Test"

descriptors:
  - name: "CarbonCount"
    function_name: "count_carbons"
    module_path: "descriptors"
    category: "constitutional"
    description: "Count carbon atoms"
  - name: "HeavyAtomCount"
    function_name: "count_heavy_atoms"
    module_path: "descriptors"
    category: "constitutional"
    description: "Count heavy atoms"
""")

        (plugin_dir / "descriptors.py").write_text("""
def count_carbons(mol):
    '''Count carbon atoms'''
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)

def count_heavy_atoms(mol):
    '''Count heavy atoms (non-hydrogen)'''
    if mol is None:
        return 0
    return mol.GetNumHeavyAtoms()
""")

        discover_plugins(paths=[tmp_path])

        # Alkane series: methane, ethane, propane, butane
        alkanes = ["C", "CC", "CCC", "CCCC"]

        carbon_func = registry.get_descriptor("CarbonCount")
        heavy_func = registry.get_descriptor("HeavyAtomCount")

        assert carbon_func is not None
        assert heavy_func is not None

        for i, smiles in enumerate(alkanes, 1):
            mol = Chem.MolFromSmiles(smiles)
            carbon_count = carbon_func(mol)
            heavy_count = heavy_func(mol)

            # For alkanes, carbon count equals the index
            assert carbon_count == i, f"Expected {i} carbons for {smiles}, got {carbon_count}"
            # Heavy atoms = carbons for alkanes
            assert heavy_count == i, f"Expected {i} heavy atoms for {smiles}, got {heavy_count}"

    def test_drug_like_filtering(self, tmp_path):
        """Test using plugin descriptors for drug-like filtering"""
        plugin_dir = tmp_path / "druglike_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "druglike_plugin"
version: "1.0.0"
author: "Test"

descriptors:
  - name: "RotatableBondRatio"
    function_name: "calc_rotatable_ratio"
    module_path: "descriptors"
    category: "constitutional"
    description: "Ratio of rotatable bonds to total bonds"
""")

        (plugin_dir / "descriptors.py").write_text("""
from rdkit.Chem import rdMolDescriptors

def calc_rotatable_ratio(mol):
    '''Calculate ratio of rotatable bonds to total bonds'''
    if mol is None:
        return 0.0
    total_bonds = mol.GetNumBonds()
    if total_bonds == 0:
        return 0.0
    rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return rotatable / total_bonds
""")

        discover_plugins(paths=[tmp_path])

        rot_ratio_func = registry.get_descriptor("RotatableBondRatio")
        assert rot_ratio_func is not None

        # Test with different molecules
        # Benzene - no rotatable bonds
        benzene = Chem.MolFromSmiles("c1ccccc1")
        assert rot_ratio_func(benzene) == 0.0

        # Butane - has rotatable bonds
        butane = Chem.MolFromSmiles("CCCC")
        ratio = rot_ratio_func(butane)
        assert ratio > 0.0

    def test_plugin_reload_scenario(self, tmp_path):
        """Test scenario where plugin might be updated and reloaded"""
        plugin_dir = tmp_path / "reload_plugin"
        plugin_dir.mkdir()

        # Initial version
        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "reload_plugin"
version: "1.0.0"
author: "Test"

descriptors:
  - name: "ReloadDesc"
    function_name: "calc_reload"
    module_path: "descriptors"
    category: "constitutional"
""")

        (plugin_dir / "descriptors.py").write_text("""
def calc_reload(mol):
    return 1.0
""")

        # First discovery
        discover_plugins(paths=[tmp_path])

        func1 = registry.get_descriptor("ReloadDesc")
        mol = Chem.MolFromSmiles("C")
        result1 = func1(mol)
        assert result1 == 1.0

        # Update plugin (in real scenario, this would be file modification)
        # For now, just verify the current state persists
        func2 = registry.get_descriptor("ReloadDesc")
        result2 = func2(mol)
        assert result2 == result1  # Should be consistent


class TestAvailabilityReport:
    """Test registry availability reporting with plugins"""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset registry and loader before and after each test"""
        clear_plugin_loader()
        clear_registry()
        yield
        clear_plugin_loader()
        clear_registry()

    def test_availability_report_includes_plugins(self, tmp_path):
        """Test that availability report includes plugin descriptors"""
        plugin_dir = tmp_path / "avail_report_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "avail_report_plugin"
version: "1.0.0"
author: "Test"

descriptors:
  - name: "AvailReportDesc"
    function_name: "calc_avail_report"
    module_path: "descriptors"
    category: "constitutional"
""")

        (plugin_dir / "descriptors.py").write_text("""
def calc_avail_report(mol):
    return 1.0
""")

        discover_plugins(paths=[tmp_path])

        report = registry.get_availability_report()

        assert "total_registered" in report
        assert report["total_registered"] > 0
        assert "by_category" in report
        assert "rdkit_version" in report


class TestDescriptorRegistrationIntegration:
    """Test DescriptorRegistration model integration"""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset registry and loader before and after each test"""
        clear_plugin_loader()
        clear_registry()
        yield
        clear_plugin_loader()
        clear_registry()

    def test_plugin_descriptor_registration_details(self, tmp_path):
        """Test that plugin descriptor registration contains correct details"""
        plugin_dir = tmp_path / "reg_detail_plugin"
        plugin_dir.mkdir()

        (plugin_dir / "plugin.yaml").write_text("""
plugin_name: "reg_detail_plugin"
version: "1.0.0"
author: "Test"

descriptors:
  - name: "RegDetailDesc"
    function_name: "calc_reg_detail"
    module_path: "descriptors"
    category: "constitutional"
""")

        (plugin_dir / "descriptors.py").write_text("""
def calc_reg_detail(mol):
    return 1.0
""")

        discover_plugins(paths=[tmp_path])

        # Get full registration info
        registration = registry.get_descriptor_registration("RegDetailDesc")

        if registration is not None:
            assert registration.name == "RegDetailDesc"
            assert registration.is_builtin is False
            assert registration.plugin_name == "reg_detail_plugin"
            assert callable(registration.function)
            assert registration.registered_at is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
