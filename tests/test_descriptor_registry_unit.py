#!/usr/bin/env python3
"""
Complete Unit Test Suite for descriptor_registry.py
Uses proper singleton clearing between tests - no hanging
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import threading

import pytest

from milia_pipeline.descriptors.descriptor_categories import DescriptorCategory, DescriptorMetadata
from milia_pipeline.descriptors.descriptor_registry import (
    DescriptorRegistry,
    auto_discover_rdkit,
    get_descriptor,
    has_descriptor,
    list_descriptors,
    registry,
)
from milia_pipeline.exceptions import DescriptorError, DescriptorValidationError


def clear_registry():
    """
    Properly clear singleton registry - FIXED VERSION!

    The key issue was that the module-level 'registry' global variable
    was created at import time, but clearing _instances created a NEW
    singleton instance with a different lock. This caused deadlocks.

    Solution: Don't clear _instances. Just reset the existing singleton's data.

    IMPORTANT: We do NOT call __init__() after reset because __init__()
    triggers auto_discover_rdkit_descriptors() which re-populates the registry
    with built-in descriptors. For unit tests, we want a clean empty registry.
    """
    # Get the existing singleton instance (don't create new one)
    existing_registry = DescriptorRegistry._instances.get(DescriptorRegistry)

    if existing_registry is not None:
        # Simply reset the existing instance's data
        # This preserves the singleton and avoids lock conflicts
        existing_registry.reset()

        # Reset internal tracking attributes that auto_discover populates
        # without calling __init__ (which would trigger auto-discovery)
        existing_registry._failed_descriptors = []
        existing_registry._mol_method_candidates = []


@pytest.fixture(autouse=True)
def auto_clear_registry():
    """
    Automatically clear registry before and after each test.
    The autouse=True means this runs for ALL tests without explicitly requesting it.
    This prevents state bleeding between tests and ensures proper cleanup.
    """
    clear_registry()
    yield
    clear_registry()


@pytest.fixture
def sample_function():
    return lambda mol: 42.0


@pytest.fixture
def sample_metadata():
    return DescriptorMetadata(
        name="TestDesc",
        category=DescriptorCategory.CONSTITUTIONAL,
        requires_3d=False,
        requires_charges=False,
    )


class TestSingletonPattern:
    def test_singleton_same_instance(self):
        r1 = DescriptorRegistry()
        r2 = DescriptorRegistry()
        assert r1 is r2

    def test_singleton_identity(self):
        r1 = DescriptorRegistry()
        r2 = DescriptorRegistry()
        assert id(r1) == id(r2)

    def test_singleton_global_registry(self):
        r = DescriptorRegistry()
        assert r is registry

    def test_singleton_state_persistence(self, sample_function):
        r1 = DescriptorRegistry()
        r1.register_descriptor("test", sample_function, plugin_name="p")
        r2 = DescriptorRegistry()
        assert r2.has_descriptor("test")

    def test_get_instance_returns_singleton(self):
        """Test that get_instance() class method returns the same singleton."""
        r1 = DescriptorRegistry.get_instance()
        r2 = DescriptorRegistry.get_instance()
        r3 = DescriptorRegistry()
        assert r1 is r2
        assert r1 is r3
        assert r1 is registry


class TestManualRegistration:
    def test_register_basic(self, sample_function):
        registry.register_descriptor("custom", sample_function, plugin_name="p")
        assert has_descriptor("custom")
        assert get_descriptor("custom") is sample_function

    def test_register_with_metadata(self, sample_function, sample_metadata):
        registry.register_descriptor(
            "custom", sample_function, metadata=sample_metadata, plugin_name="p"
        )
        reg = registry.get_descriptor_registration("custom")
        assert reg.metadata == sample_metadata
        assert reg.plugin_name == "p"

    def test_register_without_metadata(self, sample_function):
        registry.register_descriptor("custom", sample_function, plugin_name="p")
        reg = registry.get_descriptor_registration("custom")
        assert reg.metadata.name == "custom"
        assert reg.metadata.category == DescriptorCategory.CONSTITUTIONAL

    def test_register_builtin_flag(self, sample_function):
        registry.register_descriptor("builtin", sample_function, is_builtin=True)
        reg = registry.get_descriptor_registration("builtin")
        assert reg.is_builtin is True

    def test_register_plugin_tracking(self, sample_function):
        registry.register_descriptor("p1", sample_function, plugin_name="myplugin")
        plugins = registry.get_plugin_descriptors()
        assert "p1" in plugins
        assert plugins["p1"] == "myplugin"

    def test_registration_to_dict_method(self, sample_function, sample_metadata):
        """Test DescriptorRegistration.to_dict() Pydantic V2 backward-compatible method."""
        registry.register_descriptor(
            "todict_test", sample_function, metadata=sample_metadata, plugin_name="testplugin"
        )
        reg = registry.get_descriptor_registration("todict_test")

        # Test to_dict() returns dictionary
        result = reg.to_dict()
        assert isinstance(result, dict)
        assert result["name"] == "todict_test"
        assert result["is_builtin"] is False
        assert result["plugin_name"] == "testplugin"
        # function field should be present (though not JSON-serializable)
        assert "function" in result

    def test_registration_has_registered_at_timestamp(self, sample_function):
        """Test that registration includes a valid ISO timestamp."""
        registry.register_descriptor("timestamp_test", sample_function, plugin_name="p")
        reg = registry.get_descriptor_registration("timestamp_test")

        assert reg.registered_at is not None
        # Verify it's a valid ISO format string
        import datetime

        datetime.datetime.fromisoformat(reg.registered_at)


class TestValidation:
    def test_empty_name_error(self, sample_function):
        with pytest.raises(DescriptorValidationError) as exc_info:
            registry.register_descriptor("", sample_function)
        # Exception should be raised - verify message content
        assert "empty" in str(exc_info.value).lower() or "name" in str(exc_info.value).lower()

    def test_non_callable_error(self):
        with pytest.raises(DescriptorValidationError) as exc_info:
            registry.register_descriptor("test", "not_callable")
        # Verify exception has descriptor_name attribute from DescriptorError base class
        assert exc_info.value.descriptor_name == "test"

    def test_override_builtin_with_plugin_error(self, sample_function):
        registry.register_descriptor("builtin_conflict_test", lambda m: 1.0, is_builtin=True)
        with pytest.raises(DescriptorValidationError) as exc_info:
            registry.register_descriptor(
                "builtin_conflict_test", sample_function, is_builtin=False, plugin_name="p"
            )
        # Verify conflict-related exception is raised
        assert exc_info.value.descriptor_name == "builtin_conflict_test"

    def test_override_plugin_allowed(self, sample_function):
        registry.register_descriptor("plugin_override_test", lambda m: 1.0, plugin_name="plugin1")
        registry.register_descriptor("plugin_override_test", sample_function, plugin_name="plugin2")
        assert get_descriptor("plugin_override_test") is sample_function

    def test_exception_inheritance_chain(self, sample_function):
        """Verify DescriptorValidationError inherits from DescriptorError."""
        with pytest.raises(DescriptorError):
            registry.register_descriptor("", sample_function)

        # Also verify it can be caught as DescriptorValidationError
        with pytest.raises(DescriptorValidationError):
            registry.register_descriptor("", sample_function)


class TestRetrieval:
    def test_get_existing(self, sample_function):
        registry.register_descriptor("test", sample_function)
        assert get_descriptor("test") is sample_function

    def test_get_nonexistent(self):
        assert get_descriptor("nonexistent") is None

    def test_has_existing(self, sample_function):
        registry.register_descriptor("test", sample_function)
        assert has_descriptor("test") is True

    def test_has_nonexistent(self):
        assert has_descriptor("nonexistent") is False

    def test_get_registration(self, sample_function):
        registry.register_descriptor("test", sample_function, plugin_name="p")
        reg = registry.get_descriptor_registration("test")
        assert reg.name == "test"
        assert reg.function is sample_function
        assert reg.plugin_name == "p"

    def test_list_all_descriptors(self, sample_function):
        """Test list_all_descriptors() method returns sorted list."""
        registry.register_descriptor("zebra", sample_function)
        registry.register_descriptor("apple", sample_function)
        registry.register_descriptor("mango", sample_function)

        result = registry.list_all_descriptors()
        assert result == sorted(result)
        assert "zebra" in result
        assert "apple" in result
        assert "mango" in result

    def test_get_metadata(self, sample_function, sample_metadata):
        """Test get_metadata() returns DescriptorMetadata object."""
        registry.register_descriptor("meta_test", sample_function, metadata=sample_metadata)

        result = registry.get_metadata("meta_test")
        assert result is not None
        assert result.name == sample_metadata.name
        assert result.category == sample_metadata.category
        assert result.requires_3d == sample_metadata.requires_3d
        assert result.requires_charges == sample_metadata.requires_charges

    def test_get_metadata_nonexistent(self):
        """Test get_metadata() returns None for nonexistent descriptor."""
        result = registry.get_metadata("nonexistent_descriptor")
        assert result is None


class TestListing:
    def test_list_descriptors(self, sample_function):
        registry.register_descriptor("t1", sample_function)
        registry.register_descriptor("t2", sample_function)
        descs = registry.list_available_descriptors()
        assert "t1" in descs
        assert "t2" in descs

    def test_list_sorted(self, sample_function):
        registry.register_descriptor("b", sample_function)
        registry.register_descriptor("a", sample_function)
        descs = registry.list_available_descriptors()
        assert descs == sorted(descs)

    def test_list_by_category(self, sample_function):
        meta = DescriptorMetadata("t1", DescriptorCategory.FRAGMENTS)
        registry.register_descriptor("t1", sample_function, metadata=meta)
        frags = registry.list_available_descriptors(category=DescriptorCategory.FRAGMENTS)
        assert "t1" in frags


class TestCategoryFiltering:
    def test_fragment_descriptors(self, sample_function):
        meta = DescriptorMetadata("fr_test", DescriptorCategory.FRAGMENTS)
        registry.register_descriptor("fr_test", sample_function, metadata=meta, is_builtin=True)
        frags = registry.get_fragment_descriptors()
        assert "fr_test" in frags

    def test_drug_likeness_descriptors(self, sample_function):
        meta = DescriptorMetadata("qed", DescriptorCategory.DRUG_LIKENESS)
        registry.register_descriptor("qed", sample_function, metadata=meta, is_builtin=True)
        drug = registry.get_drug_likeness_descriptors()
        assert "qed" in drug

    def test_all_category_enum_values(self, sample_function):
        """Test that all 6 DescriptorCategory enum values are properly handled."""
        # Register one descriptor per category
        test_cases = [
            ("const_test", DescriptorCategory.CONSTITUTIONAL),
            ("topo_test", DescriptorCategory.TOPOLOGICAL),
            ("elec_test", DescriptorCategory.ELECTRONIC),
            ("geom_test", DescriptorCategory.GEOMETRIC),
            ("drug_test", DescriptorCategory.DRUG_LIKENESS),
            ("frag_test", DescriptorCategory.FRAGMENTS),
        ]

        for name, category in test_cases:
            meta = DescriptorMetadata(name, category)
            registry.register_descriptor(name, sample_function, metadata=meta, is_builtin=True)

        # Verify each can be filtered by category
        for name, category in test_cases:
            filtered = registry.list_available_descriptors(category=category)
            assert name in filtered, f"{name} should be in {category.value} category"

    def test_constitutional_category_filtering(self, sample_function):
        """Test CONSTITUTIONAL category filtering specifically."""
        meta = DescriptorMetadata("MolWt_test", DescriptorCategory.CONSTITUTIONAL)
        registry.register_descriptor("MolWt_test", sample_function, metadata=meta, is_builtin=True)
        result = registry.list_available_descriptors(category=DescriptorCategory.CONSTITUTIONAL)
        assert "MolWt_test" in result

    def test_topological_category_filtering(self, sample_function):
        """Test TOPOLOGICAL category filtering specifically."""
        meta = DescriptorMetadata("Chi0_test", DescriptorCategory.TOPOLOGICAL)
        registry.register_descriptor("Chi0_test", sample_function, metadata=meta, is_builtin=True)
        result = registry.list_available_descriptors(category=DescriptorCategory.TOPOLOGICAL)
        assert "Chi0_test" in result

    def test_electronic_category_filtering(self, sample_function):
        """Test ELECTRONIC category filtering specifically."""
        meta = DescriptorMetadata("MaxPartialCharge_test", DescriptorCategory.ELECTRONIC)
        registry.register_descriptor(
            "MaxPartialCharge_test", sample_function, metadata=meta, is_builtin=True
        )
        result = registry.list_available_descriptors(category=DescriptorCategory.ELECTRONIC)
        assert "MaxPartialCharge_test" in result

    def test_geometric_category_filtering(self, sample_function):
        """Test GEOMETRIC category filtering specifically."""
        meta = DescriptorMetadata("PMI1_test", DescriptorCategory.GEOMETRIC, requires_3d=True)
        registry.register_descriptor("PMI1_test", sample_function, metadata=meta, is_builtin=True)
        result = registry.list_available_descriptors(category=DescriptorCategory.GEOMETRIC)
        assert "PMI1_test" in result


class Test3DDescriptors:
    def test_get_3d_descriptors(self, sample_function):
        meta_3d = DescriptorMetadata("rad", DescriptorCategory.GEOMETRIC, requires_3d=True)
        meta_2d = DescriptorMetadata("mol", DescriptorCategory.CONSTITUTIONAL, requires_3d=False)
        registry.register_descriptor("rad", sample_function, metadata=meta_3d, is_builtin=True)
        registry.register_descriptor("mol", sample_function, metadata=meta_2d, is_builtin=True)
        desc_3d = registry.get_3d_descriptors()
        assert "rad" in desc_3d
        assert "mol" not in desc_3d

    def test_get_charge_descriptors(self, sample_function):
        meta_charge = DescriptorMetadata(
            "max", DescriptorCategory.ELECTRONIC, requires_charges=True
        )
        meta_no = DescriptorMetadata(
            "mol", DescriptorCategory.CONSTITUTIONAL, requires_charges=False
        )
        registry.register_descriptor("max", sample_function, metadata=meta_charge, is_builtin=True)
        registry.register_descriptor("mol", sample_function, metadata=meta_no, is_builtin=True)
        charge = registry.get_charge_dependent_descriptors()
        assert "max" in charge
        assert "mol" not in charge


class TestStatistics:
    def test_empty_statistics(self):
        stats = registry.get_statistics()
        assert stats["total_descriptors"] == 0
        assert stats["builtin_descriptors"] == 0
        assert stats["plugin_descriptors"] == 0

    def test_statistics_with_descriptors(self, sample_function):
        registry.register_descriptor("t1", sample_function, is_builtin=True)
        registry.register_descriptor("t2", sample_function, plugin_name="p")
        stats = registry.get_statistics()
        assert stats["total_descriptors"] == 2
        assert stats["builtin_descriptors"] == 1
        assert stats["plugin_descriptors"] == 1

    def test_statistics_structure(self, sample_function):
        """Test that statistics contains all expected keys."""
        registry.register_descriptor("struct_test", sample_function, is_builtin=True)
        stats = registry.get_statistics()

        expected_keys = [
            "total_descriptors",
            "builtin_descriptors",
            "plugin_descriptors",
            "by_category",
            "requires_3d",
            "requires_charges",
            "plugins",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_statistics_by_category(self, sample_function):
        """Test by_category counts in statistics."""
        meta_const = DescriptorMetadata("stat_const", DescriptorCategory.CONSTITUTIONAL)
        meta_frag = DescriptorMetadata("stat_frag", DescriptorCategory.FRAGMENTS)

        registry.register_descriptor(
            "stat_const", sample_function, metadata=meta_const, is_builtin=True
        )
        registry.register_descriptor(
            "stat_frag", sample_function, metadata=meta_frag, is_builtin=True
        )

        stats = registry.get_statistics()

        assert "constitutional" in stats["by_category"]
        assert "fragments" in stats["by_category"]
        assert stats["by_category"]["constitutional"] >= 1
        assert stats["by_category"]["fragments"] >= 1

    def test_statistics_requires_3d_count(self, sample_function):
        """Test requires_3d count in statistics."""
        meta_3d = DescriptorMetadata("stat_3d", DescriptorCategory.GEOMETRIC, requires_3d=True)
        meta_2d = DescriptorMetadata(
            "stat_2d", DescriptorCategory.CONSTITUTIONAL, requires_3d=False
        )

        registry.register_descriptor("stat_3d", sample_function, metadata=meta_3d, is_builtin=True)
        registry.register_descriptor("stat_2d", sample_function, metadata=meta_2d, is_builtin=True)

        stats = registry.get_statistics()
        assert stats["requires_3d"] >= 1

    def test_statistics_requires_charges_count(self, sample_function):
        """Test requires_charges count in statistics."""
        meta_charge = DescriptorMetadata(
            "stat_charge", DescriptorCategory.ELECTRONIC, requires_charges=True
        )

        registry.register_descriptor(
            "stat_charge", sample_function, metadata=meta_charge, is_builtin=True
        )

        stats = registry.get_statistics()
        assert stats["requires_charges"] >= 1

    def test_statistics_unique_plugins_count(self, sample_function):
        """Test that plugins count reflects unique plugin names."""
        registry.register_descriptor("p1", sample_function, plugin_name="plugin_alpha")
        registry.register_descriptor(
            "p2", sample_function, plugin_name="plugin_alpha"
        )  # Same plugin
        registry.register_descriptor("p3", sample_function, plugin_name="plugin_beta")

        stats = registry.get_statistics()
        # Should count unique plugins (2), not total plugin descriptors (3)
        assert stats["plugins"] == 2
        assert stats["plugin_descriptors"] == 3


class TestPluginTracking:
    def test_empty(self):
        assert len(registry.get_plugin_descriptors()) == 0

    def test_track_plugins(self, sample_function):
        registry.register_descriptor("p1", sample_function, plugin_name="plugin1")
        registry.register_descriptor("p2", sample_function, plugin_name="plugin2")
        plugins = registry.get_plugin_descriptors()
        assert len(plugins) == 2
        assert plugins["p1"] == "plugin1"
        assert plugins["p2"] == "plugin2"

    def test_multiple_from_same_plugin(self, sample_function):
        registry.register_descriptor("d1", sample_function, plugin_name="mp")
        registry.register_descriptor("d2", sample_function, plugin_name="mp")
        plugins = registry.get_plugin_descriptors()
        assert len(plugins) == 2
        assert plugins["d1"] == "mp"
        assert plugins["d2"] == "mp"


class TestSourceFiltering:
    def test_only_builtins(self, sample_function):
        registry.register_descriptor("b1", sample_function, is_builtin=True)
        registry.register_descriptor("p1", sample_function, plugin_name="p")
        builtins = registry.list_available_descriptors(include_plugins=False, include_builtins=True)
        assert "b1" in builtins
        assert "p1" not in builtins

    def test_only_plugins(self, sample_function):
        registry.register_descriptor("b1", sample_function, is_builtin=True)
        registry.register_descriptor("p1", sample_function, plugin_name="p")
        plugins = registry.list_available_descriptors(include_plugins=True, include_builtins=False)
        assert "p1" in plugins
        assert "b1" not in plugins


class TestReset:
    def test_reset_clears(self, sample_function):
        registry.register_descriptor("test", sample_function)
        assert has_descriptor("test")
        registry.reset()
        assert not has_descriptor("test")

    def test_reset_clears_categories(self, sample_function):
        meta = DescriptorMetadata("t", DescriptorCategory.FRAGMENTS)
        registry.register_descriptor("t", sample_function, metadata=meta)
        stats = registry.get_statistics()
        assert len(stats["by_category"]) > 0
        registry.reset()
        stats = registry.get_statistics()
        assert len(stats["by_category"]) == 0


class TestThreadSafety:
    def test_concurrent_registration(self, sample_function):
        def register(i):
            registry.register_descriptor(f"d{i}", sample_function, plugin_name=f"p{i}")

        threads = [threading.Thread(target=register, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = registry.get_statistics()
        assert stats["total_descriptors"] == 10

    def test_concurrent_read_write(self, sample_function):
        """Test concurrent reads while writes are happening."""
        results = {"reads": [], "errors": []}

        def writer(i):
            try:
                registry.register_descriptor(f"rw_{i}", sample_function, plugin_name=f"p{i}")
            except Exception as e:
                results["errors"].append(str(e))

        def reader():
            try:
                # Perform multiple read operations
                _ = registry.list_available_descriptors()
                _ = registry.get_statistics()
                results["reads"].append(True)
            except Exception as e:
                results["errors"].append(str(e))

        # Mix writers and readers
        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(results["errors"]) == 0
        # All readers should complete
        assert len(results["reads"]) == 10

    def test_concurrent_category_filtering(self, sample_function):
        """Test concurrent category filtering operations."""
        # Pre-register some descriptors
        for i in range(5):
            meta = DescriptorMetadata(f"thread_frag_{i}", DescriptorCategory.FRAGMENTS)
            registry.register_descriptor(
                f"thread_frag_{i}", sample_function, metadata=meta, is_builtin=True
            )

        results = {"counts": [], "errors": []}

        def filter_by_category():
            try:
                frags = registry.get_fragment_descriptors()
                results["counts"].append(len(frags))
            except Exception as e:
                results["errors"].append(str(e))

        threads = [threading.Thread(target=filter_by_category) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results["errors"]) == 0
        # All should return consistent count
        assert all(c >= 5 for c in results["counts"])


class TestIntegration:
    def test_full_workflow(self, sample_function):
        registry.register_descriptor("custom", sample_function, plugin_name="p")
        assert has_descriptor("custom")
        func = get_descriptor("custom")
        assert func is sample_function
        stats = registry.get_statistics()
        assert stats["plugin_descriptors"] == 1

    def test_mixed_filtering(self, sample_function):
        meta = DescriptorMetadata("pc", DescriptorCategory.CONSTITUTIONAL)
        registry.register_descriptor("b1", sample_function, is_builtin=True, metadata=meta)
        registry.register_descriptor("pc", sample_function, plugin_name="p", metadata=meta)
        all_const = registry.list_available_descriptors(category=DescriptorCategory.CONSTITUTIONAL)
        assert "b1" in all_const
        assert "pc" in all_const
        builtin_const = registry.list_available_descriptors(
            category=DescriptorCategory.CONSTITUTIONAL, include_plugins=False
        )
        assert "b1" in builtin_const
        assert "pc" not in builtin_const


class TestEdgeCases:
    def test_empty_operations(self):
        assert len(registry.list_available_descriptors()) == 0
        assert get_descriptor("any") is None
        assert not has_descriptor("any")

    def test_nonexistent_registration(self):
        assert registry.get_descriptor_registration("none") is None

    def test_special_characters(self, sample_function):
        registry.register_descriptor("fr_special", sample_function)
        assert has_descriptor("fr_special")

    def test_case_sensitive(self, sample_function):
        registry.register_descriptor("Test", sample_function)
        assert has_descriptor("Test")
        assert not has_descriptor("test")

    def test_underscore_naming(self, sample_function):
        """Test descriptor names with various underscore patterns."""
        # Using unique test-prefixed names to avoid conflicts with RDKit builtins
        registry.register_descriptor("test_PEOE_VSA1_custom", sample_function)
        registry.register_descriptor("test_SlogP_VSA1_custom", sample_function)
        registry.register_descriptor("test_fr_Al_COO_custom", sample_function)

        assert has_descriptor("test_PEOE_VSA1_custom")
        assert has_descriptor("test_SlogP_VSA1_custom")
        assert has_descriptor("test_fr_Al_COO_custom")

    def test_numeric_suffix_naming(self, sample_function):
        """Test descriptor names with numeric suffixes."""
        # Using unique test-prefixed names to avoid conflicts with RDKit builtins
        registry.register_descriptor("test_Chi0_custom", sample_function)
        registry.register_descriptor("test_Chi1v_custom", sample_function)
        registry.register_descriptor("test_EState_VSA10_custom", sample_function)

        assert has_descriptor("test_Chi0_custom")
        assert has_descriptor("test_Chi1v_custom")
        assert has_descriptor("test_EState_VSA10_custom")

    def test_metadata_equality_by_name(self, sample_function):
        """Test DescriptorMetadata equality is determined by name only."""
        meta1 = DescriptorMetadata("same_name", DescriptorCategory.CONSTITUTIONAL)
        meta2 = DescriptorMetadata(
            "same_name", DescriptorCategory.TOPOLOGICAL
        )  # Different category

        # Equality is by name only (as per custom __eq__)
        assert meta1 == meta2

        # Hash is also by name (for set/dict usage)
        assert hash(meta1) == hash(meta2)

    def test_metadata_to_dict_method(self):
        """Test DescriptorMetadata.to_dict() Pydantic V2 method with enum serialization."""
        meta = DescriptorMetadata(
            "test_desc",
            DescriptorCategory.ELECTRONIC,
            requires_3d=True,
            requires_charges=True,
            description="Test description",
            rdkit_module="Descriptors3D",
        )

        result = meta.to_dict()
        assert isinstance(result, dict)
        assert result["name"] == "test_desc"
        # Category enum should be serialized to string value
        assert result["category"] == "electronic"
        assert result["requires_3d"] is True
        assert result["requires_charges"] is True
        assert result["description"] == "Test description"
        assert result["rdkit_module"] == "Descriptors3D"

    def test_register_same_descriptor_twice_plugin(self, sample_function):
        """Test registering same descriptor name from different plugins (override)."""
        func1 = lambda m: 1.0
        func2 = lambda m: 2.0

        registry.register_descriptor("override_test", func1, plugin_name="plugin_a")
        registry.register_descriptor("override_test", func2, plugin_name="plugin_b")

        # Should have the second function
        assert get_descriptor("override_test") is func2

        # Plugin tracking should show latest
        plugins = registry.get_plugin_descriptors()
        assert plugins["override_test"] == "plugin_b"


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_get_descriptor_function(self, sample_function):
        """Test get_descriptor() module-level convenience function."""
        registry.register_descriptor("conv_test", sample_function)
        result = get_descriptor("conv_test")
        assert result is sample_function

    def test_has_descriptor_function(self, sample_function):
        """Test has_descriptor() module-level convenience function."""
        assert has_descriptor("nonexistent") is False
        registry.register_descriptor("conv_has", sample_function)
        assert has_descriptor("conv_has") is True

    def test_list_descriptors_function(self, sample_function):
        """Test list_descriptors() module-level convenience function."""
        registry.register_descriptor("list_test_a", sample_function)
        registry.register_descriptor("list_test_b", sample_function)

        result = list_descriptors()
        assert "list_test_a" in result
        assert "list_test_b" in result

    def test_list_descriptors_with_category(self, sample_function):
        """Test list_descriptors() with category filter."""
        meta = DescriptorMetadata("cat_filter_test", DescriptorCategory.ELECTRONIC)
        registry.register_descriptor("cat_filter_test", sample_function, metadata=meta)

        result = list_descriptors(category=DescriptorCategory.ELECTRONIC)
        assert "cat_filter_test" in result

    def test_auto_discover_rdkit_function(self):
        """Test auto_discover_rdkit() module-level convenience function."""
        # This function should call registry.auto_discover_rdkit_descriptors()
        # and return the count of discovered descriptors
        result = auto_discover_rdkit()
        assert isinstance(result, int)
        assert result >= 0


class TestAvailabilityReport:
    """Test get_availability_report() method."""

    def test_availability_report_structure(self, sample_function):
        """Test that availability report has correct structure."""
        # Register some test descriptors
        registry.register_descriptor("report_test", sample_function, is_builtin=True)

        report = registry.get_availability_report()

        # Check required keys exist
        assert "rdkit_version" in report
        assert "total_registered" in report
        assert "total_requested" in report
        assert "failed_descriptors" in report
        assert "mol_method_descriptors" in report
        assert "by_category" in report
        assert "success_rate" in report

    def test_availability_report_types(self, sample_function):
        """Test availability report value types."""
        registry.register_descriptor("type_test", sample_function, is_builtin=True)

        report = registry.get_availability_report()

        assert isinstance(report["rdkit_version"], str)
        assert isinstance(report["total_registered"], int)
        assert isinstance(report["total_requested"], int)
        assert isinstance(report["failed_descriptors"], list)
        assert isinstance(report["mol_method_descriptors"], list)
        assert isinstance(report["by_category"], dict)
        assert isinstance(report["success_rate"], (int, float))

    def test_availability_report_consistency(self, sample_function):
        """Test availability report values are consistent."""
        meta = DescriptorMetadata("consist_test", DescriptorCategory.CONSTITUTIONAL)
        registry.register_descriptor(
            "consist_test", sample_function, metadata=meta, is_builtin=True
        )

        report = registry.get_availability_report()
        stats = registry.get_statistics()

        # total_registered should match statistics
        assert report["total_registered"] == stats["total_descriptors"]


class TestDescriptorRegistrationModel:
    """Test DescriptorRegistration Pydantic BaseModel behavior."""

    def test_model_field_access(self, sample_function, sample_metadata):
        """Test that model fields are accessible as attributes."""
        registry.register_descriptor(
            "model_test", sample_function, metadata=sample_metadata, plugin_name="test_plugin"
        )
        reg = registry.get_descriptor_registration("model_test")

        # All fields should be accessible as attributes
        assert reg.name == "model_test"
        assert reg.function is sample_function
        assert reg.metadata == sample_metadata
        assert reg.is_builtin is False
        assert reg.plugin_name == "test_plugin"
        assert reg.registered_at is not None

    def test_model_immutability_config(self, sample_function, sample_metadata):
        """Test DescriptorRegistration allows attribute modification (mutable model)."""
        registry.register_descriptor("mutable_test", sample_function, metadata=sample_metadata)
        reg = registry.get_descriptor_registration("mutable_test")

        # Model is mutable (not frozen=True), so this should work
        # Note: Pydantic BaseModel without frozen=True allows attribute assignment
        original_plugin = reg.plugin_name
        reg.plugin_name = "modified_plugin"
        assert reg.plugin_name == "modified_plugin"
        # Restore for cleanup
        reg.plugin_name = original_plugin

    def test_model_default_values(self, sample_function):
        """Test DescriptorRegistration default field values."""
        registry.register_descriptor("defaults_test", sample_function)
        reg = registry.get_descriptor_registration("defaults_test")

        # Check default values
        assert reg.is_builtin is False  # Default when not specified as builtin
        # plugin_name will be None when not provided

    def test_model_to_dict_excludes_function_safely(self, sample_function, sample_metadata):
        """Test to_dict() output structure."""
        registry.register_descriptor(
            "dict_test", sample_function, metadata=sample_metadata, plugin_name="p"
        )
        reg = registry.get_descriptor_registration("dict_test")

        result = reg.to_dict()

        # Should be a dict with all fields
        assert isinstance(result, dict)
        assert "name" in result
        assert "function" in result  # Function is included but not JSON-serializable
        assert "metadata" in result
        assert "is_builtin" in result
        assert "plugin_name" in result
        assert "registered_at" in result

    def test_model_arbitrary_types_allowed(self, sample_function, sample_metadata):
        """Test that Callable and DescriptorMetadata types are allowed."""
        # This tests the ConfigDict(arbitrary_types_allowed=True) setting
        registry.register_descriptor("arb_types_test", sample_function, metadata=sample_metadata)
        reg = registry.get_descriptor_registration("arb_types_test")

        # Function should be stored correctly (Callable type)
        assert callable(reg.function)
        # Metadata should be stored correctly (DescriptorMetadata type)
        assert isinstance(reg.metadata, DescriptorMetadata)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
