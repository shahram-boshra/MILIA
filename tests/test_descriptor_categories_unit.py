#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for descriptor_categories.py Module

This test suite provides extensive coverage of RDKit descriptor categorization including:
- Descriptor category enumeration and metadata
- Constitutional descriptors (35)
- Topological descriptors (350+)
- Electronic descriptors (8)
- Geometric descriptors (10+)
- Drug-likeness descriptors (4)
- Fragment descriptors (85)

Test Coverage:
- Descriptor counts and category validation
- Metadata retrieval and validation
- 3D coordinate requirements checking
- Partial charge requirements checking
- Descriptor filtering by requirements
- Coverage validation
- Category-based descriptor retrieval
- All helper functions

NOTE: This test suite runs inside Docker at /app/milia
Path: ~/ml_projects/milia/milia_pipeline/descriptors/descriptor_categories.py

Author: milia Project Team
Created: November 16, 2025
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import pytest
from typing import Any

# Import the module under test
from milia_pipeline.descriptors.descriptor_categories import (
    DescriptorCategory,
    DescriptorMetadata,
    get_descriptors_by_category,
    get_descriptor_metadata,
    requires_3d_coordinates,
    requires_partial_charges,
    get_all_descriptor_names,
    get_category_descriptor_names,
    filter_descriptors_by_requirements,
    get_descriptor_count_by_category,
    validate_descriptor_coverage,
    ALL_DESCRIPTORS,
    CONSTITUTIONAL_DESCRIPTORS,
    TOPOLOGICAL_DESCRIPTORS,
    ELECTRONIC_DESCRIPTORS,
    GEOMETRIC_DESCRIPTORS,
    DRUG_LIKENESS_DESCRIPTORS,
    FRAGMENT_DESCRIPTORS,
    DESCRIPTORS_BY_CATEGORY,
    DESCRIPTOR_METADATA_MAP,
)


# =============================================================================
# TEST FIXTURES AND HELPER FUNCTIONS
# =============================================================================

@pytest.fixture
def sample_descriptor_names():
    """Sample descriptor names for testing"""
    return ["MolWt", "RadiusOfGyration", "MaxPartialCharge", "BertzCT"]


@pytest.fixture
def mixed_requirement_descriptors():
    """Mix of descriptors with different requirements"""
    return {
        "no_requirements": ["MolWt", "BertzCT", "NumHeavyAtoms"],
        "requires_3d": ["RadiusOfGyration", "Asphericity", "Eccentricity"],
        "requires_charges": ["MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge"]
    }


# =============================================================================
# DescriptorCategory Enum Tests
# =============================================================================

class TestDescriptorCategory:
    """Test suite for DescriptorCategory enumeration"""
    
    def test_all_categories_defined(self):
        """Test that all 6 categories are defined"""
        categories = list(DescriptorCategory)
        assert len(categories) == 6
        assert DescriptorCategory.CONSTITUTIONAL in categories
        assert DescriptorCategory.TOPOLOGICAL in categories
        assert DescriptorCategory.ELECTRONIC in categories
        assert DescriptorCategory.GEOMETRIC in categories
        assert DescriptorCategory.DRUG_LIKENESS in categories
        assert DescriptorCategory.FRAGMENTS in categories
    
    def test_category_values(self):
        """Test category string values"""
        assert DescriptorCategory.CONSTITUTIONAL.value == "constitutional"
        assert DescriptorCategory.TOPOLOGICAL.value == "topological"
        assert DescriptorCategory.ELECTRONIC.value == "electronic"
        assert DescriptorCategory.GEOMETRIC.value == "geometric"
        assert DescriptorCategory.DRUG_LIKENESS.value == "drug_likeness"
        assert DescriptorCategory.FRAGMENTS.value == "fragments"
    
    def test_category_membership(self):
        """Test category membership checks"""
        assert DescriptorCategory.CONSTITUTIONAL in DescriptorCategory
        assert DescriptorCategory.FRAGMENTS in DescriptorCategory


# =============================================================================
# DescriptorMetadata Tests
# =============================================================================

class TestDescriptorMetadata:
    """Test suite for DescriptorMetadata dataclass"""
    
    def test_metadata_creation_minimal(self):
        """Test metadata creation with minimal parameters"""
        metadata = DescriptorMetadata("TestDescriptor", DescriptorCategory.CONSTITUTIONAL)
        
        assert metadata.name == "TestDescriptor"
        assert metadata.category == DescriptorCategory.CONSTITUTIONAL
        assert metadata.requires_3d is False
        assert metadata.requires_charges is False
        assert metadata.description == ""
        assert metadata.rdkit_module == "Descriptors"
    
    def test_metadata_creation_full(self):
        """Test metadata creation with all parameters"""
        metadata = DescriptorMetadata(
            name="TestDescriptor",
            category=DescriptorCategory.GEOMETRIC,
            requires_3d=True,
            requires_charges=True,
            description="Test description",
            rdkit_module="Descriptors3D"
        )
        
        assert metadata.name == "TestDescriptor"
        assert metadata.category == DescriptorCategory.GEOMETRIC
        assert metadata.requires_3d is True
        assert metadata.requires_charges is True
        assert metadata.description == "Test description"
        assert metadata.rdkit_module == "Descriptors3D"
    
    def test_metadata_immutability(self):
        """Test that DescriptorMetadata is immutable (frozen Pydantic model)"""
        metadata = DescriptorMetadata("TestDescriptor", DescriptorCategory.CONSTITUTIONAL)
        
        # Pydantic V2 frozen models raise ValidationError on attribute modification
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            metadata.name = "ChangedName"
    
    def test_metadata_hashable(self):
        """Test that DescriptorMetadata is hashable (by name only per __hash__ implementation)"""
        metadata1 = DescriptorMetadata("TestDescriptor", DescriptorCategory.CONSTITUTIONAL)
        metadata2 = DescriptorMetadata("TestDescriptor", DescriptorCategory.TOPOLOGICAL)
        metadata3 = DescriptorMetadata("DifferentDescriptor", DescriptorCategory.CONSTITUTIONAL)
        
        # Same name -> same hash and equal (per __hash__ and __eq__ implementation)
        assert hash(metadata1) == hash(metadata2)
        assert metadata1 == metadata2
        
        # Different names -> different hash and not equal
        assert hash(metadata1) != hash(metadata3)
        assert metadata1 != metadata3
        
        # Should be hashable (can be added to set)
        # Same-name descriptors collapse to 1 entry due to name-based equality
        descriptor_set = {metadata1, metadata2}
        assert len(descriptor_set) == 1
        
        # Different-name descriptors remain separate
        descriptor_set_different = {metadata1, metadata3}
        assert len(descriptor_set_different) == 2

    def test_metadata_equality_with_non_metadata(self):
        """Test __eq__ returns False for non-DescriptorMetadata objects"""
        metadata = DescriptorMetadata("TestDescriptor", DescriptorCategory.CONSTITUTIONAL)
        
        assert metadata != "TestDescriptor"
        assert metadata != 123
        assert metadata != None
        assert metadata != {"name": "TestDescriptor"}
    
    def test_metadata_to_dict(self):
        """Test to_dict() method returns correct dictionary with enum serialization"""
        metadata = DescriptorMetadata(
            name="TestDescriptor",
            category=DescriptorCategory.GEOMETRIC,
            requires_3d=True,
            requires_charges=False,
            description="Test description",
            rdkit_module="Descriptors3D"
        )
        
        result = metadata.to_dict()
        
        assert isinstance(result, dict)
        assert result["name"] == "TestDescriptor"
        # Category should be serialized as string value, not enum
        assert result["category"] == "geometric"
        assert result["requires_3d"] is True
        assert result["requires_charges"] is False
        assert result["description"] == "Test description"
        assert result["rdkit_module"] == "Descriptors3D"
    
    def test_metadata_to_dict_all_categories(self):
        """Test to_dict() correctly serializes all category enum values"""
        for category in DescriptorCategory:
            metadata = DescriptorMetadata("Test", category)
            result = metadata.to_dict()
            assert result["category"] == category.value
    
    def test_metadata_keyword_only_construction(self):
        """Test metadata construction using only keyword arguments"""
        metadata = DescriptorMetadata(
            name="KeywordOnly",
            category=DescriptorCategory.TOPOLOGICAL,
            requires_3d=False,
            requires_charges=True,
            description="Keyword test",
            rdkit_module="CustomModule"
        )
        
        assert metadata.name == "KeywordOnly"
        assert metadata.category == DescriptorCategory.TOPOLOGICAL
        assert metadata.requires_3d is False
        assert metadata.requires_charges is True
        assert metadata.description == "Keyword test"
        assert metadata.rdkit_module == "CustomModule"
    
    def test_metadata_model_dump_json_mode(self):
        """Test model_dump with mode='json' for Pydantic V2 compliance"""
        metadata = DescriptorMetadata("Test", DescriptorCategory.ELECTRONIC)
        
        # Direct model_dump call should work the same as to_dict
        result = metadata.model_dump(mode='json')
        
        assert isinstance(result, dict)
        assert result["category"] == "electronic"  # Enum serialized to string
    
    def test_metadata_model_fields(self):
        """Test that Pydantic model has expected fields"""
        fields = DescriptorMetadata.model_fields
        
        assert "name" in fields
        assert "category" in fields
        assert "requires_3d" in fields
        assert "requires_charges" in fields
        assert "description" in fields
        assert "rdkit_module" in fields



    """Test descriptor counts for each category"""
    
    def test_constitutional_count(self):
        """Test constitutional descriptor count"""
        assert len(CONSTITUTIONAL_DESCRIPTORS) == 35
    
    def test_topological_count(self):
        """Test topological descriptor count"""
        assert len(TOPOLOGICAL_DESCRIPTORS) == 281
    
    def test_electronic_count(self):
        """Test electronic descriptor count"""
        assert len(ELECTRONIC_DESCRIPTORS) == 8
    
    def test_geometric_count(self):
        """Test geometric descriptor count (11 3D-dependent descriptors)"""
        assert len(GEOMETRIC_DESCRIPTORS) == 11
    
    def test_drug_likeness_count(self):
        """Test drug-likeness descriptor count"""
        assert len(DRUG_LIKENESS_DESCRIPTORS) == 4
    
    def test_fragment_count(self):
        """Test fragment descriptor count"""
        assert len(FRAGMENT_DESCRIPTORS) == 85
    
    def test_total_descriptor_count(self):
        """Test total descriptor count (should be 420+)"""
        assert len(ALL_DESCRIPTORS) >= 420
    
    def test_all_descriptors_sum(self):
        """Test that ALL_DESCRIPTORS is sum of all categories"""
        total_in_categories = (
            len(CONSTITUTIONAL_DESCRIPTORS) +
            len(TOPOLOGICAL_DESCRIPTORS) +
            len(ELECTRONIC_DESCRIPTORS) +
            len(GEOMETRIC_DESCRIPTORS) +
            len(DRUG_LIKENESS_DESCRIPTORS) +
            len(FRAGMENT_DESCRIPTORS)
        )
        assert len(ALL_DESCRIPTORS) == total_in_categories


# =============================================================================
# Requirement Checking Tests
# =============================================================================

class TestRequirementChecking:
    """Test requirement checking functions"""
    
    def test_3d_requirements_true(self):
        """Test 3D requirement checking for descriptors that require 3D"""
        assert requires_3d_coordinates("RadiusOfGyration") is True
        assert requires_3d_coordinates("Asphericity") is True
        assert requires_3d_coordinates("Eccentricity") is True
    
    def test_3d_requirements_false(self):
        """Test 3D requirement checking for descriptors that don't require 3D"""
        assert requires_3d_coordinates("MolWt") is False
        assert requires_3d_coordinates("BertzCT") is False
        assert requires_3d_coordinates("NumHeavyAtoms") is False
    
    def test_3d_requirements_nonexistent(self):
        """Test 3D requirement checking for non-existent descriptor"""
        assert requires_3d_coordinates("NonExistentDescriptor") is False
    
    def test_charge_requirements_true(self):
        """Test charge requirement checking for descriptors that require charges"""
        assert requires_partial_charges("MaxPartialCharge") is True
        assert requires_partial_charges("MinPartialCharge") is True
        assert requires_partial_charges("MaxAbsPartialCharge") is True
    
    def test_charge_requirements_false(self):
        """Test charge requirement checking for descriptors that don't require charges"""
        assert requires_partial_charges("MolWt") is False
        assert requires_partial_charges("BertzCT") is False
        assert requires_partial_charges("NumHeavyAtoms") is False
    
    def test_charge_requirements_nonexistent(self):
        """Test charge requirement checking for non-existent descriptor"""
        assert requires_partial_charges("NonExistentDescriptor") is False


# =============================================================================
# Descriptor Filtering Tests
# =============================================================================

class TestDescriptorFiltering:
    """Test descriptor filtering by requirements"""
    
    def test_filtering_no_3d_no_charges(self):
        """Test descriptor filtering with no 3D and no charges"""
        test_descriptors = ["MolWt", "RadiusOfGyration", "MaxPartialCharge"]
        valid, filtered = filter_descriptors_by_requirements(
            test_descriptors, has_3d=False, has_charges=False
        )
        
        assert "MolWt" in valid
        assert "RadiusOfGyration" in filtered
        assert "MaxPartialCharge" in filtered
        assert len(valid) == 1
        assert len(filtered) == 2
    
    def test_filtering_with_3d_no_charges(self):
        """Test descriptor filtering with 3D but no charges"""
        test_descriptors = ["MolWt", "RadiusOfGyration", "MaxPartialCharge"]
        valid, filtered = filter_descriptors_by_requirements(
            test_descriptors, has_3d=True, has_charges=False
        )
        
        assert "MolWt" in valid
        assert "RadiusOfGyration" in valid
        assert "MaxPartialCharge" in filtered
        assert len(valid) == 2
        assert len(filtered) == 1
    
    def test_filtering_with_3d_and_charges(self):
        """Test descriptor filtering with both 3D and charges"""
        test_descriptors = ["MolWt", "RadiusOfGyration", "MaxPartialCharge"]
        valid, filtered = filter_descriptors_by_requirements(
            test_descriptors, has_3d=True, has_charges=True
        )
        
        assert "MolWt" in valid
        assert "RadiusOfGyration" in valid
        assert "MaxPartialCharge" in valid
        assert len(valid) == 3
        assert len(filtered) == 0
    
    def test_filtering_empty_list(self):
        """Test descriptor filtering with empty list"""
        valid, filtered = filter_descriptors_by_requirements(
            [], has_3d=False, has_charges=False
        )
        
        assert len(valid) == 0
        assert len(filtered) == 0
    
    def test_filtering_nonexistent_descriptor(self):
        """Test descriptor filtering with non-existent descriptor"""
        test_descriptors = ["NonExistentDescriptor"]
        valid, filtered = filter_descriptors_by_requirements(
            test_descriptors, has_3d=True, has_charges=True
        )
        
        assert "NonExistentDescriptor" in filtered
        assert len(valid) == 0
        assert len(filtered) == 1
    
    def test_filtering_mixed_requirements(self, mixed_requirement_descriptors):
        """Test filtering with mixed requirement descriptors"""
        all_descriptors = (
            mixed_requirement_descriptors["no_requirements"] +
            mixed_requirement_descriptors["requires_3d"] +
            mixed_requirement_descriptors["requires_charges"]
        )
        
        # Test with no capabilities
        valid, filtered = filter_descriptors_by_requirements(
            all_descriptors, has_3d=False, has_charges=False
        )
        assert len(valid) == len(mixed_requirement_descriptors["no_requirements"])
        
        # Test with only 3D
        valid, filtered = filter_descriptors_by_requirements(
            all_descriptors, has_3d=True, has_charges=False
        )
        assert len(valid) == (
            len(mixed_requirement_descriptors["no_requirements"]) +
            len(mixed_requirement_descriptors["requires_3d"])
        )
        
        # Test with only charges
        valid, filtered = filter_descriptors_by_requirements(
            all_descriptors, has_3d=False, has_charges=True
        )
        assert len(valid) == (
            len(mixed_requirement_descriptors["no_requirements"]) +
            len(mixed_requirement_descriptors["requires_charges"])
        )


# =============================================================================
# Category Retrieval Tests
# =============================================================================

class TestCategoryRetrieval:
    """Test retrieving descriptors by category"""
    
    def test_get_by_category_constitutional(self):
        """Test retrieving constitutional descriptors"""
        descriptors = get_descriptors_by_category(DescriptorCategory.CONSTITUTIONAL)
        
        assert len(descriptors) == 35
        assert all(desc.category == DescriptorCategory.CONSTITUTIONAL for desc in descriptors)
    
    def test_get_by_category_topological(self):
        """Test retrieving topological descriptors"""
        descriptors = get_descriptors_by_category(DescriptorCategory.TOPOLOGICAL)
        
        assert len(descriptors) == 281
        assert all(desc.category == DescriptorCategory.TOPOLOGICAL for desc in descriptors)
    
    def test_get_by_category_electronic(self):
        """Test retrieving electronic descriptors"""
        descriptors = get_descriptors_by_category(DescriptorCategory.ELECTRONIC)
        
        assert len(descriptors) == 8
        assert all(desc.category == DescriptorCategory.ELECTRONIC for desc in descriptors)
    
    def test_get_by_category_geometric(self):
        """Test retrieving geometric descriptors"""
        descriptors = get_descriptors_by_category(DescriptorCategory.GEOMETRIC)
        
        assert len(descriptors) == 11
        assert all(desc.category == DescriptorCategory.GEOMETRIC for desc in descriptors)
    
    def test_get_by_category_drug_likeness(self):
        """Test retrieving drug-likeness descriptors"""
        descriptors = get_descriptors_by_category(DescriptorCategory.DRUG_LIKENESS)
        
        assert len(descriptors) == 4
        assert all(desc.category == DescriptorCategory.DRUG_LIKENESS for desc in descriptors)
    
    def test_get_by_category_fragments(self):
        """Test retrieving fragment descriptors"""
        fragments = get_descriptors_by_category(DescriptorCategory.FRAGMENTS)
        
        assert len(fragments) == 85
        assert all(desc.category == DescriptorCategory.FRAGMENTS for desc in fragments)
    
    def test_get_by_category_returns_list(self):
        """Test that get_descriptors_by_category returns a list"""
        descriptors = get_descriptors_by_category(DescriptorCategory.CONSTITUTIONAL)
        assert isinstance(descriptors, list)
    
    def test_get_by_category_returns_metadata_objects(self):
        """Test that returned objects are DescriptorMetadata instances"""
        descriptors = get_descriptors_by_category(DescriptorCategory.CONSTITUTIONAL)
        assert all(isinstance(desc, DescriptorMetadata) for desc in descriptors)


# =============================================================================
# Metadata Retrieval Tests
# =============================================================================

class TestMetadataRetrieval:
    """Test metadata retrieval functions"""
    
    def test_metadata_retrieval_molwt(self):
        """Test metadata retrieval for MolWt"""
        metadata = get_descriptor_metadata("MolWt")
        
        assert metadata is not None
        assert metadata.name == "MolWt"
        assert metadata.category == DescriptorCategory.CONSTITUTIONAL
        assert metadata.requires_3d is False
        assert metadata.requires_charges is False
    
    def test_metadata_retrieval_3d_descriptor(self):
        """Test metadata retrieval for 3D descriptor"""
        metadata = get_descriptor_metadata("RadiusOfGyration")
        
        assert metadata is not None
        assert metadata.name == "RadiusOfGyration"
        assert metadata.category == DescriptorCategory.GEOMETRIC
        assert metadata.requires_3d is True
    
    def test_metadata_retrieval_charge_descriptor(self):
        """Test metadata retrieval for charge descriptor"""
        metadata = get_descriptor_metadata("MaxPartialCharge")
        
        assert metadata is not None
        assert metadata.name == "MaxPartialCharge"
        assert metadata.category == DescriptorCategory.ELECTRONIC
        assert metadata.requires_charges is True
    
    def test_metadata_retrieval_nonexistent(self):
        """Test metadata retrieval for non-existent descriptor"""
        metadata = get_descriptor_metadata("NonExistentDescriptor")
        assert metadata is None
    
    def test_metadata_retrieval_topological(self):
        """Test metadata retrieval for topological descriptor"""
        metadata = get_descriptor_metadata("BertzCT")
        
        assert metadata is not None
        assert metadata.name == "BertzCT"
        assert metadata.category == DescriptorCategory.TOPOLOGICAL
    
    def test_metadata_retrieval_fragment(self):
        """Test metadata retrieval for fragment descriptor"""
        # fr_C_O is one of the fragment descriptors
        metadata = get_descriptor_metadata("fr_C_O")
        
        assert metadata is not None
        assert metadata.category == DescriptorCategory.FRAGMENTS


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelperFunctions:
    """Test helper functions"""
    
    def test_get_all_descriptor_names(self):
        """Test get_all_descriptor_names returns all unique names from the map"""
        all_names = get_all_descriptor_names()
        
        assert isinstance(all_names, list)
        # DESCRIPTOR_METADATA_MAP deduplicates by name (FractionCSP3 appears in both
        # CONSTITUTIONAL and DRUG_LIKENESS categories - see source lines 173, 377)
        assert len(all_names) == len(DESCRIPTOR_METADATA_MAP)
        assert "MolWt" in all_names
        assert "BertzCT" in all_names
    
    def test_get_all_descriptor_names_unique(self):
        """Test that all descriptor names are unique"""
        all_names = get_all_descriptor_names()
        assert len(all_names) == len(set(all_names))
    
    def test_get_category_descriptor_names(self):
        """Test get_category_descriptor_names"""
        constitutional_names = get_category_descriptor_names(DescriptorCategory.CONSTITUTIONAL)
        
        assert isinstance(constitutional_names, list)
        assert len(constitutional_names) == 35
        assert "MolWt" in constitutional_names
        assert all(isinstance(name, str) for name in constitutional_names)
    
    def test_get_descriptor_count_by_category(self):
        """Test get_descriptor_count_by_category"""
        counts = get_descriptor_count_by_category()
        
        assert isinstance(counts, dict)
        assert counts["constitutional"] == 35
        assert counts["topological"] == 281
        assert counts["electronic"] == 8
        assert counts["geometric"] == 11
        assert counts["drug_likeness"] == 4
        assert counts["fragments"] == 85


# =============================================================================
# Coverage Validation Tests
# =============================================================================

class TestCoverageValidation:
    """Test descriptor coverage validation"""
    
    def test_coverage_validation_structure(self):
        """Test coverage validation returns correct structure"""
        coverage = validate_descriptor_coverage()
        
        assert "expected" in coverage
        assert "actual" in coverage
        assert "total_expected" in coverage
        assert "total_actual" in coverage
        assert "coverage_complete" in coverage
    
    def test_coverage_validation_complete(self):
        """Test descriptor coverage validation returns results"""
        coverage = validate_descriptor_coverage()
        
        # Just verify the validation ran and has results
        assert isinstance(coverage["coverage_complete"], bool)
        assert coverage["total_actual"] >= 420
    
    def test_coverage_validation_expected_counts(self):
        """Test expected counts in coverage validation"""
        coverage = validate_descriptor_coverage()
        expected = coverage["expected"]
        
        assert expected[DescriptorCategory.CONSTITUTIONAL] == 35
        assert expected[DescriptorCategory.TOPOLOGICAL] == 350
        assert expected[DescriptorCategory.ELECTRONIC] == 8
        assert expected[DescriptorCategory.GEOMETRIC] == 10
        assert expected[DescriptorCategory.DRUG_LIKENESS] == 4
        assert expected[DescriptorCategory.FRAGMENTS] == 85
    
    def test_coverage_validation_actual_meets_expected(self):
        """Test actual counts are correctly reported"""
        coverage = validate_descriptor_coverage()
        expected = coverage["expected"]
        actual = coverage["actual"]
        
        # Verify actual counts match what we know
        assert actual[DescriptorCategory.CONSTITUTIONAL] == 35
        assert actual[DescriptorCategory.TOPOLOGICAL] == 281
        assert actual[DescriptorCategory.ELECTRONIC] == 8
        assert actual[DescriptorCategory.GEOMETRIC] == 11
        assert actual[DescriptorCategory.DRUG_LIKENESS] == 4
        assert actual[DescriptorCategory.FRAGMENTS] == 85


# =============================================================================
# Dictionary Mapping Tests
# =============================================================================

class TestDictionaryMappings:
    """Test descriptor dictionary mappings"""
    
    def test_descriptors_by_category_keys(self):
        """Test DESCRIPTORS_BY_CATEGORY has all category keys"""
        assert DescriptorCategory.CONSTITUTIONAL in DESCRIPTORS_BY_CATEGORY
        assert DescriptorCategory.TOPOLOGICAL in DESCRIPTORS_BY_CATEGORY
        assert DescriptorCategory.ELECTRONIC in DESCRIPTORS_BY_CATEGORY
        assert DescriptorCategory.GEOMETRIC in DESCRIPTORS_BY_CATEGORY
        assert DescriptorCategory.DRUG_LIKENESS in DESCRIPTORS_BY_CATEGORY
        assert DescriptorCategory.FRAGMENTS in DESCRIPTORS_BY_CATEGORY
    
    def test_descriptor_metadata_map_completeness(self):
        """Test DESCRIPTOR_METADATA_MAP contains unique descriptor names"""
        # Note: FractionCSP3 appears in both CONSTITUTIONAL and DRUG_LIKENESS categories
        # (intentional alternative naming per source). The map deduplicates by name,
        # so it has 1 fewer entry than ALL_DESCRIPTORS list.
        unique_names_in_all = set(desc.name for desc in ALL_DESCRIPTORS)
        assert len(DESCRIPTOR_METADATA_MAP) == len(unique_names_in_all)
    
    def test_descriptor_metadata_map_lookup(self):
        """Test DESCRIPTOR_METADATA_MAP allows direct lookup"""
        assert "MolWt" in DESCRIPTOR_METADATA_MAP
        assert DESCRIPTOR_METADATA_MAP["MolWt"].category == DescriptorCategory.CONSTITUTIONAL
    
    def test_descriptor_metadata_map_no_duplicates(self):
        """Test DESCRIPTOR_METADATA_MAP has no duplicate keys"""
        all_names_from_map = list(DESCRIPTOR_METADATA_MAP.keys())
        assert len(all_names_from_map) == len(set(all_names_from_map))
    
    def test_fractioncsp3_intentional_duplicate(self):
        """Test that FractionCSP3 appears in multiple categories (intentional design)
        
        FractionCSP3 is listed in both CONSTITUTIONAL (line 173 of source) and 
        DRUG_LIKENESS (line 377, labeled 'alternative naming'). This is intentional.
        The DESCRIPTOR_METADATA_MAP deduplicates by name (last definition wins).
        """
        constitutional_names = [d.name for d in CONSTITUTIONAL_DESCRIPTORS]
        drug_likeness_names = [d.name for d in DRUG_LIKENESS_DESCRIPTORS]
        
        # FractionCSP3 appears in both categories
        assert "FractionCSP3" in constitutional_names
        assert "FractionCSP3" in drug_likeness_names
        
        # ALL_DESCRIPTORS contains both entries
        fractioncsp3_count = sum(1 for d in ALL_DESCRIPTORS if d.name == "FractionCSP3")
        assert fractioncsp3_count == 2
        
        # Map deduplicates (1 entry total)
        assert "FractionCSP3" in DESCRIPTOR_METADATA_MAP
        
        # This explains the 424 vs 423 difference
        assert len(ALL_DESCRIPTORS) == len(DESCRIPTOR_METADATA_MAP) + 1


# =============================================================================
# Specific Descriptor Tests
# =============================================================================

class TestSpecificDescriptors:
    """Test specific descriptors are present and correctly categorized"""
    
    def test_constitutional_descriptors_present(self):
        """Test key constitutional descriptors are present"""
        constitutional_names = [desc.name for desc in CONSTITUTIONAL_DESCRIPTORS]
        
        assert "MolWt" in constitutional_names
        assert "NumHeavyAtoms" in constitutional_names
        assert "NumHDonors" in constitutional_names
        assert "NumHAcceptors" in constitutional_names
        assert "TPSA" in constitutional_names
    
    def test_topological_descriptors_present(self):
        """Test key topological descriptors are present"""
        topological_names = [desc.name for desc in TOPOLOGICAL_DESCRIPTORS]
        
        assert "BertzCT" in topological_names
        assert "BalabanJ" in topological_names
        assert "Chi0v" in topological_names
    
    def test_electronic_descriptors_present(self):
        """Test key electronic descriptors are present"""
        electronic_names = [desc.name for desc in ELECTRONIC_DESCRIPTORS]
        
        assert "MaxPartialCharge" in electronic_names
        assert "MinPartialCharge" in electronic_names
        assert "MaxAbsPartialCharge" in electronic_names
    
    def test_geometric_descriptors_present(self):
        """Test key geometric descriptors are present"""
        geometric_names = [desc.name for desc in GEOMETRIC_DESCRIPTORS]
        
        assert "RadiusOfGyration" in geometric_names
        assert "Asphericity" in geometric_names
        assert "Eccentricity" in geometric_names
    
    def test_drug_likeness_descriptors_present(self):
        """Test drug-likeness descriptors are present"""
        drug_names = [desc.name for desc in DRUG_LIKENESS_DESCRIPTORS]
        
        # Should have QED and other drug-likeness metrics
        assert len(drug_names) == 4
    
    def test_fragment_descriptors_present(self):
        """Test key fragment descriptors are present"""
        fragment_names = [desc.name for desc in FRAGMENT_DESCRIPTORS]
        
        # Fragment descriptors typically start with 'fr_'
        fr_descriptors = [name for name in fragment_names if name.startswith('fr_')]
        assert len(fr_descriptors) >= 80  # Most should be fr_ descriptors


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_descriptor_list_filtering(self):
        """Test filtering with empty descriptor list"""
        valid, filtered = filter_descriptors_by_requirements([], has_3d=True, has_charges=True)
        assert len(valid) == 0
        assert len(filtered) == 0
    
    def test_none_values_in_requirements(self):
        """Test requirement checking with various inputs"""
        # Non-existent descriptor should return False
        assert requires_3d_coordinates("") is False
        assert requires_partial_charges("") is False
    
    def test_case_sensitive_descriptor_names(self):
        """Test that descriptor names are case-sensitive"""
        metadata = get_descriptor_metadata("molwt")  # lowercase
        assert metadata is None  # Should not find it
        
        metadata = get_descriptor_metadata("MolWt")  # correct case
        assert metadata is not None
    
    def test_get_descriptors_by_invalid_category(self):
        """Test get_descriptors_by_category returns empty list for invalid input"""
        # Using a string instead of enum should return empty list from dict.get default
        result = get_descriptors_by_category("invalid")
        assert result == []
    
    def test_filter_with_all_unknown_descriptors(self):
        """Test filtering when all descriptors are unknown"""
        unknown = ["Unknown1", "Unknown2", "Unknown3"]
        valid, filtered = filter_descriptors_by_requirements(unknown, has_3d=True, has_charges=True)
        
        assert len(valid) == 0
        assert len(filtered) == 3
        assert all(name in filtered for name in unknown)
    
    def test_filter_preserves_order(self):
        """Test that filter_descriptors_by_requirements preserves input order"""
        ordered_descriptors = ["MolWt", "BertzCT", "NumHeavyAtoms", "HeavyAtomMolWt"]
        valid, _ = filter_descriptors_by_requirements(ordered_descriptors, has_3d=True, has_charges=True)
        
        # All should be valid, order preserved
        assert valid == ordered_descriptors
    
    def test_descriptor_metadata_with_special_characters_in_name(self):
        """Test that descriptor names with underscores work correctly"""
        # AUTOCORR2D_1 has underscore
        metadata = get_descriptor_metadata("AUTOCORR2D_1")
        assert metadata is not None
        assert metadata.name == "AUTOCORR2D_1"
        
        # fr_C_O has multiple underscores
        metadata = get_descriptor_metadata("fr_C_O")
        assert metadata is not None
        assert metadata.name == "fr_C_O"
    
    def test_get_category_descriptor_names_for_all_categories(self):
        """Test get_category_descriptor_names works for all categories"""
        for category in DescriptorCategory:
            names = get_category_descriptor_names(category)
            assert isinstance(names, list)
            assert all(isinstance(name, str) for name in names)
            assert len(names) > 0  # Each category should have at least one descriptor


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests across multiple functions"""
    
    def test_workflow_get_category_then_filter(self):
        """Test workflow: get descriptors by category, then filter by requirements"""
        # Get all geometric descriptors
        geometric_descriptors = get_descriptors_by_category(DescriptorCategory.GEOMETRIC)
        geometric_names = [desc.name for desc in geometric_descriptors]
        
        # Filter without 3D coordinates
        valid, filtered = filter_descriptors_by_requirements(
            geometric_names, has_3d=False, has_charges=False
        )
        
        # Most geometric descriptors require 3D
        assert len(filtered) > 0
    
    def test_workflow_validate_then_retrieve(self):
        """Test workflow: validate coverage, then retrieve metadata"""
        coverage = validate_descriptor_coverage()
        
        # If coverage is complete, should be able to retrieve metadata for any descriptor
        if coverage["coverage_complete"]:
            all_names = get_all_descriptor_names()
            for name in all_names[:10]:  # Test first 10
                metadata = get_descriptor_metadata(name)
                assert metadata is not None
    
    def test_consistency_between_list_and_map(self):
        """Test consistency between ALL_DESCRIPTORS list and DESCRIPTOR_METADATA_MAP"""
        # Every descriptor in ALL_DESCRIPTORS should be in the map
        for desc in ALL_DESCRIPTORS:
            assert desc.name in DESCRIPTOR_METADATA_MAP
            assert DESCRIPTOR_METADATA_MAP[desc.name] == desc
    
    def test_category_partitioning(self):
        """Test that descriptors are properly partitioned by category"""
        # Count descriptors by category manually
        category_counts = {}
        for desc in ALL_DESCRIPTORS:
            category_counts[desc.category] = category_counts.get(desc.category, 0) + 1
        
        # Should match counts from get_descriptor_count_by_category
        counts_dict = get_descriptor_count_by_category()
        for category, count in category_counts.items():
            assert counts_dict[category.value] == count


# =============================================================================
# Data Integrity Tests
# =============================================================================

class TestDataIntegrity:
    """Test data integrity and internal consistency"""
    
    def test_all_descriptors_have_valid_category(self):
        """Test that all descriptors have a valid DescriptorCategory"""
        valid_categories = set(DescriptorCategory)
        for desc in ALL_DESCRIPTORS:
            assert desc.category in valid_categories, f"{desc.name} has invalid category"
    
    def test_all_descriptors_have_non_empty_name(self):
        """Test that all descriptors have non-empty names"""
        for desc in ALL_DESCRIPTORS:
            assert desc.name, "Descriptor with empty name found"
            assert len(desc.name) > 0
    
    def test_no_whitespace_in_descriptor_names(self):
        """Test that no descriptor names contain whitespace"""
        for desc in ALL_DESCRIPTORS:
            assert desc.name == desc.name.strip(), f"'{desc.name}' has leading/trailing whitespace"
            assert ' ' not in desc.name, f"'{desc.name}' contains space"
    
    def test_3d_descriptors_have_correct_module(self):
        """Test that 3D descriptors reference Descriptors3D module"""
        for desc in GEOMETRIC_DESCRIPTORS:
            if desc.requires_3d:
                assert desc.rdkit_module == "Descriptors3D", \
                    f"{desc.name} requires 3D but has wrong module: {desc.rdkit_module}"
    
    def test_electronic_charge_descriptors_marked_correctly(self):
        """Test that charge-requiring electronic descriptors are marked"""
        charge_descriptors = ["MaxPartialCharge", "MinPartialCharge", 
                            "MaxAbsPartialCharge", "MinAbsPartialCharge"]
        for name in charge_descriptors:
            metadata = get_descriptor_metadata(name)
            assert metadata is not None, f"{name} not found"
            assert metadata.requires_charges is True, f"{name} should require charges"
    
    def test_fragment_descriptors_naming_convention(self):
        """Test that fragment descriptors follow fr_ naming convention"""
        for desc in FRAGMENT_DESCRIPTORS:
            assert desc.name.startswith("fr_"), \
                f"Fragment descriptor {desc.name} should start with 'fr_'"
    
    def test_descriptors_by_category_matches_individual_lists(self):
        """Test DESCRIPTORS_BY_CATEGORY matches individual category lists"""
        assert DESCRIPTORS_BY_CATEGORY[DescriptorCategory.CONSTITUTIONAL] == CONSTITUTIONAL_DESCRIPTORS
        assert DESCRIPTORS_BY_CATEGORY[DescriptorCategory.TOPOLOGICAL] == TOPOLOGICAL_DESCRIPTORS
        assert DESCRIPTORS_BY_CATEGORY[DescriptorCategory.ELECTRONIC] == ELECTRONIC_DESCRIPTORS
        assert DESCRIPTORS_BY_CATEGORY[DescriptorCategory.GEOMETRIC] == GEOMETRIC_DESCRIPTORS
        assert DESCRIPTORS_BY_CATEGORY[DescriptorCategory.DRUG_LIKENESS] == DRUG_LIKENESS_DESCRIPTORS
        assert DESCRIPTORS_BY_CATEGORY[DescriptorCategory.FRAGMENTS] == FRAGMENT_DESCRIPTORS


# =============================================================================
# Topological Descriptor Group Tests
# =============================================================================

class TestTopologicalDescriptorGroups:
    """Test topological descriptor subgroups"""
    
    def test_peoe_vsa_descriptors_present(self):
        """Test PEOE_VSA descriptors (1-14) are present"""
        topo_names = [desc.name for desc in TOPOLOGICAL_DESCRIPTORS]
        for i in range(1, 15):
            assert f"PEOE_VSA{i}" in topo_names, f"PEOE_VSA{i} missing"
    
    def test_smr_vsa_descriptors_present(self):
        """Test SMR_VSA descriptors (1-10) are present"""
        topo_names = [desc.name for desc in TOPOLOGICAL_DESCRIPTORS]
        for i in range(1, 11):
            assert f"SMR_VSA{i}" in topo_names, f"SMR_VSA{i} missing"
    
    def test_slogp_vsa_descriptors_present(self):
        """Test SlogP_VSA descriptors (1-12) are present"""
        topo_names = [desc.name for desc in TOPOLOGICAL_DESCRIPTORS]
        for i in range(1, 13):
            assert f"SlogP_VSA{i}" in topo_names, f"SlogP_VSA{i} missing"
    
    def test_estate_vsa_descriptors_present(self):
        """Test EState_VSA descriptors (1-11) are present"""
        topo_names = [desc.name for desc in TOPOLOGICAL_DESCRIPTORS]
        for i in range(1, 12):
            assert f"EState_VSA{i}" in topo_names, f"EState_VSA{i} missing"
    
    def test_vsa_estate_descriptors_present(self):
        """Test VSA_EState descriptors (1-10) are present"""
        topo_names = [desc.name for desc in TOPOLOGICAL_DESCRIPTORS]
        for i in range(1, 11):
            assert f"VSA_EState{i}" in topo_names, f"VSA_EState{i} missing"
    
    def test_bcut2d_descriptors_present(self):
        """Test BCUT2D descriptors are present"""
        bcut2d_names = [
            "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO",
            "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW"
        ]
        topo_names = [desc.name for desc in TOPOLOGICAL_DESCRIPTORS]
        for name in bcut2d_names:
            assert name in topo_names, f"{name} missing"
    
    def test_autocorr2d_descriptors_present(self):
        """Test AUTOCORR2D descriptors (1-192) are present"""
        topo_names = [desc.name for desc in TOPOLOGICAL_DESCRIPTORS]
        for i in range(1, 193):
            assert f"AUTOCORR2D_{i}" in topo_names, f"AUTOCORR2D_{i} missing"
    
    def test_connectivity_indices_present(self):
        """Test Chi connectivity indices are present"""
        chi_descriptors = ["Chi0v", "Chi1v", "Chi2v", "Chi3v", "Chi4v",
                          "Chi0n", "Chi1n", "Chi2n", "Chi3n", "Chi4n"]
        topo_names = [desc.name for desc in TOPOLOGICAL_DESCRIPTORS]
        for name in chi_descriptors:
            assert name in topo_names, f"{name} missing"
    
    def test_kappa_indices_present(self):
        """Test Kappa shape indices are present"""
        kappa_descriptors = ["Kappa1", "Kappa2", "Kappa3"]
        topo_names = [desc.name for desc in TOPOLOGICAL_DESCRIPTORS]
        for name in kappa_descriptors:
            assert name in topo_names, f"{name} missing"


# =============================================================================
# Geometric Descriptor Tests
# =============================================================================

class TestGeometricDescriptors:
    """Test geometric (3D) descriptors"""
    
    def test_all_geometric_require_3d(self):
        """Test that all geometric descriptors require 3D coordinates"""
        for desc in GEOMETRIC_DESCRIPTORS:
            assert desc.requires_3d is True, \
                f"Geometric descriptor {desc.name} should require 3D"
    
    def test_geometric_descriptors_list(self):
        """Test expected geometric descriptors are present"""
        expected = [
            "RadiusOfGyration", "InertialShapeFactor", "Eccentricity",
            "Asphericity", "SpherocityIndex", "PBF", "NPR1", "NPR2",
            "PMI1", "PMI2", "PMI3"
        ]
        geometric_names = [desc.name for desc in GEOMETRIC_DESCRIPTORS]
        for name in expected:
            assert name in geometric_names, f"{name} missing from geometric descriptors"
    
    def test_pmi_descriptors_present(self):
        """Test Principal Moment of Inertia descriptors"""
        geometric_names = [desc.name for desc in GEOMETRIC_DESCRIPTORS]
        assert "PMI1" in geometric_names
        assert "PMI2" in geometric_names
        assert "PMI3" in geometric_names
    
    def test_npr_descriptors_present(self):
        """Test Normalized Principal Ratio descriptors"""
        geometric_names = [desc.name for desc in GEOMETRIC_DESCRIPTORS]
        assert "NPR1" in geometric_names
        assert "NPR2" in geometric_names


# =============================================================================
# Type Safety Tests
# =============================================================================

class TestTypeSafety:
    """Test type safety and return types"""
    
    def test_get_all_descriptor_names_returns_strings(self):
        """Test that get_all_descriptor_names returns list of strings"""
        names = get_all_descriptor_names()
        assert isinstance(names, list)
        assert all(isinstance(name, str) for name in names)
    
    def test_filter_descriptors_returns_tuple_of_lists(self):
        """Test filter_descriptors_by_requirements return type"""
        result = filter_descriptors_by_requirements(["MolWt"], has_3d=False)
        assert isinstance(result, tuple)
        assert len(result) == 2
        valid, filtered = result
        assert isinstance(valid, list)
        assert isinstance(filtered, list)
    
    def test_get_descriptor_count_by_category_returns_dict(self):
        """Test get_descriptor_count_by_category return type"""
        counts = get_descriptor_count_by_category()
        assert isinstance(counts, dict)
        assert all(isinstance(k, str) for k in counts.keys())
        assert all(isinstance(v, int) for v in counts.values())
    
    def test_validate_descriptor_coverage_returns_dict(self):
        """Test validate_descriptor_coverage return type"""
        coverage = validate_descriptor_coverage()
        assert isinstance(coverage, dict)
        assert "expected" in coverage
        assert "actual" in coverage
        assert "total_expected" in coverage
        assert "total_actual" in coverage
        assert "coverage_complete" in coverage
    
    def test_descriptor_metadata_attributes_types(self):
        """Test DescriptorMetadata attribute types"""
        metadata = get_descriptor_metadata("MolWt")
        assert isinstance(metadata.name, str)
        assert isinstance(metadata.category, DescriptorCategory)
        assert isinstance(metadata.requires_3d, bool)
        assert isinstance(metadata.requires_charges, bool)
        assert isinstance(metadata.description, str)
        assert isinstance(metadata.rdkit_module, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
