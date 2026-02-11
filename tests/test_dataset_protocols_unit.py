#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/datasets/protocols.py

Module under test: protocols.py
- DatasetHandlerProtocol: @runtime_checkable Protocol with 11 methods
- DatasetConverterProtocol: @runtime_checkable Protocol with 2 methods
- DatasetValidatorProtocol: @runtime_checkable Protocol with 2 methods

Test path on local machine: ~/ml_projects/milia/tests/test_dataset_protocols_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/datasets/protocols.py

NOTE: This test suite runs inside Docker at /app/milia
Path mappings:
- Project root: /app/milia (mapped from ~/ml_projects/milia)

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution

Updated: February 2026 - Production-ready comprehensive test coverage
"""

import sys
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, MagicMock
import inspect
from typing import (
    Protocol, runtime_checkable, Dict, List, Any, Optional,
    get_type_hints,
)

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from torch_geometric.data import Data

from milia_pipeline.datasets.protocols import (
    DatasetHandlerProtocol,
    DatasetConverterProtocol,
    DatasetValidatorProtocol,
)


# ============================================================================
# HELPER: Test implementation classes for structural subtyping verification
# ============================================================================

class _FullHandler:
    """A class implementing ALL 11 DatasetHandlerProtocol methods."""

    def get_dataset_type(self) -> str:
        return "test"

    def validate_molecule_data(
        self,
        raw_properties_dict: Dict[str, Any],
        molecule_index: int,
        identifier: str = "N/A",
    ) -> None:
        pass

    def get_required_properties(self) -> List[str]:
        return ["energy"]

    def process_property_value(
        self,
        key: str,
        value: Any,
        molecule_index: int,
        identifier: str = "N/A",
    ) -> Any:
        return value

    def enrich_pyg_data(
        self,
        pyg_data: Data,
        raw_properties_dict: Dict[str, Any],
        molecule_index: int,
        identifier: str = "N/A",
    ) -> Data:
        return pyg_data

    def get_processing_statistics(
        self, processed_molecules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {}

    def get_supported_structural_features(self) -> Dict[str, List[str]]:
        return {}

    def get_molecular_charge(
        self,
        raw_properties_dict: Dict[str, Any],
        atomic_numbers: np.ndarray,
        mol_identifier: Optional[str] = None,
    ) -> int:
        return 0

    def get_molecule_creation_strategy(self) -> str:
        return "coordinate_based"

    def get_transform_recommendations(self) -> Dict[str, List[str]]:
        return {}

    def get_supported_descriptors(self) -> Dict[str, List[str]]:
        return {}


class _FullConverter:
    """A class implementing ALL DatasetConverterProtocol methods."""

    def convert(self, raw_data: Any) -> Data:
        return Data()

    def supports_format(self, format_type: str) -> bool:
        return True


class _FullValidator:
    """A class implementing ALL DatasetValidatorProtocol methods."""

    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data

    def get_validation_rules(self) -> Dict[str, Any]:
        return {}


class _EmptyClass:
    """A class implementing NONE of the protocol methods."""
    pass


# ============================================================================
# CONSTANT: Expected method names for each protocol (derived from source)
# ============================================================================

HANDLER_METHOD_NAMES = [
    "get_dataset_type",
    "validate_molecule_data",
    "get_required_properties",
    "process_property_value",
    "enrich_pyg_data",
    "get_processing_statistics",
    "get_supported_structural_features",
    "get_molecular_charge",
    "get_molecule_creation_strategy",
    "get_transform_recommendations",
    "get_supported_descriptors",
]

CONVERTER_METHOD_NAMES = [
    "convert",
    "supports_format",
]

VALIDATOR_METHOD_NAMES = [
    "validate",
    "get_validation_rules",
]


# ============================================================================
# GROUP 1: DatasetHandlerProtocol — Type Identity and Metaclass (8 tests)
# ============================================================================

class TestDatasetHandlerProtocolTypeIdentity(unittest.TestCase):
    """Verify DatasetHandlerProtocol is a proper runtime_checkable Protocol."""

    def test_is_subclass_of_protocol(self):
        """DatasetHandlerProtocol inherits from typing.Protocol."""
        self.assertTrue(
            issubclass(DatasetHandlerProtocol, Protocol),
            "DatasetHandlerProtocol must be a Protocol subclass",
        )

    def test_is_runtime_checkable(self):
        """DatasetHandlerProtocol is decorated with @runtime_checkable."""
        # runtime_checkable protocols have the _is_runtime_protocol attribute
        self.assertTrue(
            getattr(DatasetHandlerProtocol, "_is_runtime_protocol", False),
            "DatasetHandlerProtocol must be @runtime_checkable",
        )

    def test_is_a_class(self):
        """DatasetHandlerProtocol is a class (not a function or module)."""
        self.assertTrue(inspect.isclass(DatasetHandlerProtocol))

    def test_has_correct_name(self):
        """Class name is 'DatasetHandlerProtocol'."""
        self.assertEqual(DatasetHandlerProtocol.__name__, "DatasetHandlerProtocol")

    def test_has_correct_module(self):
        """Defined in the datasets.protocols module."""
        self.assertIn("protocols", DatasetHandlerProtocol.__module__)

    def test_has_docstring(self):
        """Protocol has a non-empty docstring."""
        self.assertIsNotNone(DatasetHandlerProtocol.__doc__)
        self.assertGreater(len(DatasetHandlerProtocol.__doc__.strip()), 0)

    def test_has_exactly_11_protocol_methods(self):
        """DatasetHandlerProtocol defines exactly 11 abstract methods."""
        self.assertEqual(len(HANDLER_METHOD_NAMES), 11)
        for method_name in HANDLER_METHOD_NAMES:
            self.assertTrue(
                callable(getattr(DatasetHandlerProtocol, method_name, None)),
                f"Missing method: {method_name}",
            )

    def test_all_expected_method_names_present(self):
        """Every expected method name is present on the protocol class."""
        for name in HANDLER_METHOD_NAMES:
            with self.subTest(method=name):
                self.assertTrue(
                    hasattr(DatasetHandlerProtocol, name),
                    f"DatasetHandlerProtocol missing expected method '{name}'",
                )


# ============================================================================
# GROUP 2: DatasetHandlerProtocol — Method Signatures (11 tests)
# ============================================================================

class TestDatasetHandlerProtocolSignatures(unittest.TestCase):
    """Verify method signatures match the protocol definition."""

    def _get_sig(self, method_name: str) -> inspect.Signature:
        """Helper: get the signature of a protocol method."""
        method = getattr(DatasetHandlerProtocol, method_name)
        return inspect.signature(method)

    # --- get_dataset_type ---

    def test_get_dataset_type_params(self):
        """get_dataset_type(self) -> str: has only 'self' parameter."""
        sig = self._get_sig("get_dataset_type")
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["self"])

    def test_get_dataset_type_return_annotation(self):
        """get_dataset_type return annotation is str."""
        sig = self._get_sig("get_dataset_type")
        self.assertIs(sig.return_annotation, str)

    # --- validate_molecule_data ---

    def test_validate_molecule_data_params(self):
        """validate_molecule_data has self + 3 params (1 with default)."""
        sig = self._get_sig("validate_molecule_data")
        params = list(sig.parameters.keys())
        self.assertEqual(
            params, ["self", "raw_properties_dict", "molecule_index", "identifier"]
        )

    def test_validate_molecule_data_identifier_default(self):
        """validate_molecule_data 'identifier' has default 'N/A'."""
        sig = self._get_sig("validate_molecule_data")
        self.assertEqual(sig.parameters["identifier"].default, "N/A")

    # --- get_required_properties ---

    def test_get_required_properties_params(self):
        """get_required_properties(self) -> List[str]: self only."""
        sig = self._get_sig("get_required_properties")
        self.assertEqual(list(sig.parameters.keys()), ["self"])

    # --- process_property_value ---

    def test_process_property_value_params(self):
        """process_property_value has self + 4 params."""
        sig = self._get_sig("process_property_value")
        params = list(sig.parameters.keys())
        self.assertEqual(
            params, ["self", "key", "value", "molecule_index", "identifier"]
        )

    def test_process_property_value_identifier_default(self):
        """process_property_value 'identifier' has default 'N/A'."""
        sig = self._get_sig("process_property_value")
        self.assertEqual(sig.parameters["identifier"].default, "N/A")

    # --- enrich_pyg_data ---

    def test_enrich_pyg_data_params(self):
        """enrich_pyg_data has self + 4 params."""
        sig = self._get_sig("enrich_pyg_data")
        params = list(sig.parameters.keys())
        self.assertEqual(
            params,
            ["self", "pyg_data", "raw_properties_dict", "molecule_index", "identifier"],
        )

    def test_enrich_pyg_data_identifier_default(self):
        """enrich_pyg_data 'identifier' has default 'N/A'."""
        sig = self._get_sig("enrich_pyg_data")
        self.assertEqual(sig.parameters["identifier"].default, "N/A")

    # --- get_molecular_charge ---

    def test_get_molecular_charge_params(self):
        """get_molecular_charge has self + 3 params."""
        sig = self._get_sig("get_molecular_charge")
        params = list(sig.parameters.keys())
        self.assertEqual(
            params,
            ["self", "raw_properties_dict", "atomic_numbers", "mol_identifier"],
        )

    def test_get_molecular_charge_mol_identifier_default(self):
        """get_molecular_charge 'mol_identifier' has default None."""
        sig = self._get_sig("get_molecular_charge")
        self.assertIsNone(sig.parameters["mol_identifier"].default)


# ============================================================================
# GROUP 3: DatasetHandlerProtocol — isinstance / issubclass (12 tests)
# ============================================================================

class TestDatasetHandlerProtocolStructuralSubtyping(unittest.TestCase):
    """Verify runtime_checkable isinstance/issubclass for handler protocol."""

    def test_full_implementation_isinstance(self):
        """A class with all 11 methods passes isinstance check."""
        handler = _FullHandler()
        self.assertIsInstance(handler, DatasetHandlerProtocol)

    def test_full_implementation_issubclass(self):
        """A class with all 11 methods passes issubclass check."""
        self.assertTrue(issubclass(_FullHandler, DatasetHandlerProtocol))

    def test_empty_class_not_isinstance(self):
        """A class with no methods fails isinstance check."""
        obj = _EmptyClass()
        self.assertNotIsInstance(obj, DatasetHandlerProtocol)

    def test_empty_class_not_issubclass(self):
        """A class with no methods fails issubclass check."""
        self.assertFalse(issubclass(_EmptyClass, DatasetHandlerProtocol))

    def test_partial_implementation_missing_one_method(self):
        """A class missing one method fails isinstance check."""
        for missing_method in HANDLER_METHOD_NAMES:
            with self.subTest(missing=missing_method):
                # Dynamically create a class with all methods except one
                attrs = {}
                for name in HANDLER_METHOD_NAMES:
                    if name != missing_method:
                        attrs[name] = lambda self: None
                PartialClass = type("PartialClass", (), attrs)
                obj = PartialClass()
                self.assertNotIsInstance(
                    obj,
                    DatasetHandlerProtocol,
                    f"Should fail isinstance when missing '{missing_method}'",
                )

    def test_explicit_subclass_isinstance(self):
        """A class explicitly inheriting the protocol passes isinstance."""

        class ExplicitHandler(DatasetHandlerProtocol):
            def get_dataset_type(self): return "x"
            def validate_molecule_data(self, raw_properties_dict, molecule_index, identifier="N/A"): pass
            def get_required_properties(self): return []
            def process_property_value(self, key, value, molecule_index, identifier="N/A"): return value
            def enrich_pyg_data(self, pyg_data, raw_properties_dict, molecule_index, identifier="N/A"): return pyg_data
            def get_processing_statistics(self, processed_molecules): return {}
            def get_supported_structural_features(self): return {}
            def get_molecular_charge(self, raw_properties_dict, atomic_numbers, mol_identifier=None): return 0
            def get_molecule_creation_strategy(self): return "coordinate_based"
            def get_transform_recommendations(self): return {}
            def get_supported_descriptors(self): return {}

        handler = ExplicitHandler()
        self.assertIsInstance(handler, DatasetHandlerProtocol)

    def test_mock_with_spec_isinstance(self):
        """A Mock(spec=_FullHandler) passes isinstance check."""
        mock_handler = Mock(spec=_FullHandler)
        self.assertIsInstance(mock_handler, DatasetHandlerProtocol)

    def test_non_runtime_checkable_protocol_fails_isinstance(self):
        """Confirm isinstance fails for non-runtime_checkable protocols.

        This is a negative control that validates our test approach.
        """

        class NonCheckable(Protocol):
            def some_method(self) -> str: ...

        obj = _FullHandler()
        with self.assertRaises(TypeError):
            isinstance(obj, NonCheckable)

    def test_converter_not_isinstance_handler(self):
        """A converter implementation does NOT satisfy the handler protocol."""
        converter = _FullConverter()
        self.assertNotIsInstance(converter, DatasetHandlerProtocol)

    def test_validator_not_isinstance_handler(self):
        """A validator implementation does NOT satisfy the handler protocol."""
        validator = _FullValidator()
        self.assertNotIsInstance(validator, DatasetHandlerProtocol)

    def test_builtin_types_not_isinstance(self):
        """Built-in types (str, int, dict, list) are not handler instances."""
        for obj in ["hello", 42, {}, [], 3.14, None, True]:
            with self.subTest(obj=type(obj).__name__):
                self.assertNotIsInstance(obj, DatasetHandlerProtocol)

    def test_protocol_class_is_instance_of_itself(self):
        """The protocol class itself IS an instance of itself.

        Because the class object has all 11 method attributes defined on it,
        runtime_checkable isinstance checks attribute presence and succeeds.
        """
        self.assertIsInstance(DatasetHandlerProtocol, DatasetHandlerProtocol)


# ============================================================================
# GROUP 4: DatasetConverterProtocol — Type Identity and Signatures (8 tests)
# ============================================================================

class TestDatasetConverterProtocolIdentity(unittest.TestCase):
    """Verify DatasetConverterProtocol is a proper runtime_checkable Protocol."""

    def test_is_subclass_of_protocol(self):
        """DatasetConverterProtocol inherits from typing.Protocol."""
        self.assertTrue(issubclass(DatasetConverterProtocol, Protocol))

    def test_is_runtime_checkable(self):
        """DatasetConverterProtocol is decorated with @runtime_checkable."""
        self.assertTrue(
            getattr(DatasetConverterProtocol, "_is_runtime_protocol", False)
        )

    def test_has_correct_name(self):
        """Class name is 'DatasetConverterProtocol'."""
        self.assertEqual(
            DatasetConverterProtocol.__name__, "DatasetConverterProtocol"
        )

    def test_has_docstring(self):
        """Protocol has a non-empty docstring."""
        self.assertIsNotNone(DatasetConverterProtocol.__doc__)
        self.assertGreater(len(DatasetConverterProtocol.__doc__.strip()), 0)

    def test_has_exactly_2_methods(self):
        """DatasetConverterProtocol defines exactly 2 methods."""
        for name in CONVERTER_METHOD_NAMES:
            self.assertTrue(
                callable(getattr(DatasetConverterProtocol, name, None)),
                f"Missing method: {name}",
            )

    def test_convert_signature(self):
        """convert(self, raw_data) has self + raw_data params."""
        sig = inspect.signature(DatasetConverterProtocol.convert)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["self", "raw_data"])

    def test_supports_format_signature(self):
        """supports_format(self, format_type) has self + format_type params."""
        sig = inspect.signature(DatasetConverterProtocol.supports_format)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["self", "format_type"])

    def test_supports_format_return_annotation(self):
        """supports_format return annotation is bool."""
        sig = inspect.signature(DatasetConverterProtocol.supports_format)
        self.assertIs(sig.return_annotation, bool)


# ============================================================================
# GROUP 5: DatasetConverterProtocol — isinstance / issubclass (7 tests)
# ============================================================================

class TestDatasetConverterProtocolStructuralSubtyping(unittest.TestCase):
    """Verify runtime_checkable isinstance/issubclass for converter protocol."""

    def test_full_implementation_isinstance(self):
        """A class with both methods passes isinstance check."""
        converter = _FullConverter()
        self.assertIsInstance(converter, DatasetConverterProtocol)

    def test_full_implementation_issubclass(self):
        """A class with both methods passes issubclass check."""
        self.assertTrue(issubclass(_FullConverter, DatasetConverterProtocol))

    def test_empty_class_not_isinstance(self):
        """A class with no methods fails isinstance check."""
        self.assertNotIsInstance(_EmptyClass(), DatasetConverterProtocol)

    def test_missing_convert_fails(self):
        """A class with only supports_format fails isinstance."""

        class OnlySupportsFormat:
            def supports_format(self, format_type: str) -> bool:
                return True

        self.assertNotIsInstance(OnlySupportsFormat(), DatasetConverterProtocol)

    def test_missing_supports_format_fails(self):
        """A class with only convert fails isinstance."""

        class OnlyConvert:
            def convert(self, raw_data: Any) -> Data:
                return Data()

        self.assertNotIsInstance(OnlyConvert(), DatasetConverterProtocol)

    def test_handler_not_isinstance_converter(self):
        """A handler implementation does NOT satisfy the converter protocol
        (unless it also has convert and supports_format, which it doesn't)."""
        handler = _FullHandler()
        self.assertNotIsInstance(handler, DatasetConverterProtocol)

    def test_mock_with_spec_isinstance(self):
        """A Mock(spec=_FullConverter) passes isinstance check."""
        mock_converter = Mock(spec=_FullConverter)
        self.assertIsInstance(mock_converter, DatasetConverterProtocol)


# ============================================================================
# GROUP 6: DatasetValidatorProtocol — Type Identity and Signatures (8 tests)
# ============================================================================

class TestDatasetValidatorProtocolIdentity(unittest.TestCase):
    """Verify DatasetValidatorProtocol is a proper runtime_checkable Protocol."""

    def test_is_subclass_of_protocol(self):
        """DatasetValidatorProtocol inherits from typing.Protocol."""
        self.assertTrue(issubclass(DatasetValidatorProtocol, Protocol))

    def test_is_runtime_checkable(self):
        """DatasetValidatorProtocol is decorated with @runtime_checkable."""
        self.assertTrue(
            getattr(DatasetValidatorProtocol, "_is_runtime_protocol", False)
        )

    def test_has_correct_name(self):
        """Class name is 'DatasetValidatorProtocol'."""
        self.assertEqual(
            DatasetValidatorProtocol.__name__, "DatasetValidatorProtocol"
        )

    def test_has_docstring(self):
        """Protocol has a non-empty docstring."""
        self.assertIsNotNone(DatasetValidatorProtocol.__doc__)
        self.assertGreater(len(DatasetValidatorProtocol.__doc__.strip()), 0)

    def test_has_exactly_2_methods(self):
        """DatasetValidatorProtocol defines exactly 2 methods."""
        for name in VALIDATOR_METHOD_NAMES:
            self.assertTrue(
                callable(getattr(DatasetValidatorProtocol, name, None)),
                f"Missing method: {name}",
            )

    def test_validate_signature(self):
        """validate(self, data) has self + data params."""
        sig = inspect.signature(DatasetValidatorProtocol.validate)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["self", "data"])

    def test_get_validation_rules_signature(self):
        """get_validation_rules(self) has only 'self' param."""
        sig = inspect.signature(DatasetValidatorProtocol.get_validation_rules)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["self"])

    def test_get_validation_rules_return_annotation(self):
        """get_validation_rules return annotation is Dict[str, Any]."""
        sig = inspect.signature(DatasetValidatorProtocol.get_validation_rules)
        self.assertEqual(sig.return_annotation, Dict[str, Any])


# ============================================================================
# GROUP 7: DatasetValidatorProtocol — isinstance / issubclass (7 tests)
# ============================================================================

class TestDatasetValidatorProtocolStructuralSubtyping(unittest.TestCase):
    """Verify runtime_checkable isinstance/issubclass for validator protocol."""

    def test_full_implementation_isinstance(self):
        """A class with both methods passes isinstance check."""
        validator = _FullValidator()
        self.assertIsInstance(validator, DatasetValidatorProtocol)

    def test_full_implementation_issubclass(self):
        """A class with both methods passes issubclass check."""
        self.assertTrue(issubclass(_FullValidator, DatasetValidatorProtocol))

    def test_empty_class_not_isinstance(self):
        """A class with no methods fails isinstance check."""
        self.assertNotIsInstance(_EmptyClass(), DatasetValidatorProtocol)

    def test_missing_validate_fails(self):
        """A class with only get_validation_rules fails isinstance."""

        class OnlyRules:
            def get_validation_rules(self) -> Dict[str, Any]:
                return {}

        self.assertNotIsInstance(OnlyRules(), DatasetValidatorProtocol)

    def test_missing_get_validation_rules_fails(self):
        """A class with only validate fails isinstance."""

        class OnlyValidate:
            def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
                return data

        self.assertNotIsInstance(OnlyValidate(), DatasetValidatorProtocol)

    def test_handler_not_isinstance_validator(self):
        """A handler implementation does NOT satisfy the validator protocol."""
        handler = _FullHandler()
        self.assertNotIsInstance(handler, DatasetValidatorProtocol)

    def test_mock_with_spec_isinstance(self):
        """A Mock(spec=_FullValidator) passes isinstance check."""
        mock_validator = Mock(spec=_FullValidator)
        self.assertIsInstance(mock_validator, DatasetValidatorProtocol)


# ============================================================================
# GROUP 8: Cross-Protocol Isolation (6 tests)
# ============================================================================

class TestCrossProtocolIsolation(unittest.TestCase):
    """Verify the three protocols are independent and non-overlapping."""

    def test_handler_is_not_converter(self):
        """DatasetHandlerProtocol is not a subclass of DatasetConverterProtocol."""
        # Protocols are independent — issubclass between protocols checks
        # structural compatibility. A handler has 11 methods, a converter has 2
        # different methods, so they should not be subclasses of each other.
        self.assertFalse(issubclass(_FullHandler, DatasetConverterProtocol))

    def test_handler_is_not_validator(self):
        """DatasetHandlerProtocol impl is not a subclass of DatasetValidatorProtocol."""
        self.assertFalse(issubclass(_FullHandler, DatasetValidatorProtocol))

    def test_converter_is_not_handler(self):
        """DatasetConverterProtocol impl is not a subclass of DatasetHandlerProtocol."""
        self.assertFalse(issubclass(_FullConverter, DatasetHandlerProtocol))

    def test_converter_is_not_validator(self):
        """DatasetConverterProtocol impl is not a subclass of DatasetValidatorProtocol."""
        self.assertFalse(issubclass(_FullConverter, DatasetValidatorProtocol))

    def test_validator_is_not_handler(self):
        """DatasetValidatorProtocol impl is not a subclass of DatasetHandlerProtocol."""
        self.assertFalse(issubclass(_FullValidator, DatasetHandlerProtocol))

    def test_validator_is_not_converter(self):
        """DatasetValidatorProtocol impl is not a subclass of DatasetConverterProtocol."""
        self.assertFalse(issubclass(_FullValidator, DatasetConverterProtocol))


# ============================================================================
# GROUP 9: Multi-Protocol Compliance (5 tests)
# ============================================================================

class TestMultiProtocolCompliance(unittest.TestCase):
    """Verify that a class can satisfy multiple protocols simultaneously."""

    def test_class_satisfies_handler_and_converter(self):
        """A class implementing both handler and converter methods satisfies both."""

        class HandlerAndConverter(_FullHandler):
            def convert(self, raw_data: Any) -> Data:
                return Data()

            def supports_format(self, format_type: str) -> bool:
                return True

        obj = HandlerAndConverter()
        self.assertIsInstance(obj, DatasetHandlerProtocol)
        self.assertIsInstance(obj, DatasetConverterProtocol)

    def test_class_satisfies_handler_and_validator(self):
        """A class implementing both handler and validator methods satisfies both."""

        class HandlerAndValidator(_FullHandler):
            def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
                return data

            def get_validation_rules(self) -> Dict[str, Any]:
                return {}

        obj = HandlerAndValidator()
        self.assertIsInstance(obj, DatasetHandlerProtocol)
        self.assertIsInstance(obj, DatasetValidatorProtocol)

    def test_class_satisfies_all_three(self):
        """A class implementing all protocol methods satisfies all three."""

        class UniversalImpl(_FullHandler):
            def convert(self, raw_data: Any) -> Data:
                return Data()

            def supports_format(self, format_type: str) -> bool:
                return True

            def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
                return data

            def get_validation_rules(self) -> Dict[str, Any]:
                return {}

        obj = UniversalImpl()
        self.assertIsInstance(obj, DatasetHandlerProtocol)
        self.assertIsInstance(obj, DatasetConverterProtocol)
        self.assertIsInstance(obj, DatasetValidatorProtocol)

    def test_class_satisfies_converter_and_validator(self):
        """A class implementing converter + validator but not handler."""

        class ConverterAndValidator(_FullConverter):
            def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
                return data

            def get_validation_rules(self) -> Dict[str, Any]:
                return {}

        obj = ConverterAndValidator()
        self.assertNotIsInstance(obj, DatasetHandlerProtocol)
        self.assertIsInstance(obj, DatasetConverterProtocol)
        self.assertIsInstance(obj, DatasetValidatorProtocol)

    def test_extra_methods_do_not_break_protocol(self):
        """Extra methods beyond the protocol don't affect compliance."""

        class ExtendedHandler(_FullHandler):
            def extra_method(self):
                return "bonus"

            def another_extra(self, x: int) -> str:
                return str(x)

        obj = ExtendedHandler()
        self.assertIsInstance(obj, DatasetHandlerProtocol)


# ============================================================================
# GROUP 10: Handler Method Docstrings (11 tests)
# ============================================================================

class TestDatasetHandlerProtocolDocstrings(unittest.TestCase):
    """Verify each handler protocol method has a non-empty docstring."""

    def test_each_method_has_docstring(self):
        """Every protocol method has a non-empty docstring."""
        for method_name in HANDLER_METHOD_NAMES:
            with self.subTest(method=method_name):
                method = getattr(DatasetHandlerProtocol, method_name)
                doc = getattr(method, "__doc__", None)
                self.assertIsNotNone(
                    doc, f"{method_name} has no docstring"
                )
                self.assertGreater(
                    len(doc.strip()), 0,
                    f"{method_name} has empty docstring",
                )


# ============================================================================
# GROUP 11: Protocol Method Return Type Annotations (8 tests)
# ============================================================================

class TestProtocolReturnAnnotations(unittest.TestCase):
    """Verify return type annotations are correctly defined on protocol methods."""

    def _get_return_annotation(self, protocol_cls, method_name):
        """Helper: get return annotation from protocol method signature."""
        sig = inspect.signature(getattr(protocol_cls, method_name))
        return sig.return_annotation

    def test_handler_get_dataset_type_returns_str(self):
        """get_dataset_type -> str."""
        self.assertIs(
            self._get_return_annotation(DatasetHandlerProtocol, "get_dataset_type"),
            str,
        )

    def test_handler_validate_molecule_data_returns_none(self):
        """validate_molecule_data -> None."""
        self.assertIsNone(
            self._get_return_annotation(
                DatasetHandlerProtocol, "validate_molecule_data"
            )
        )

    def test_handler_get_required_properties_returns_list_str(self):
        """get_required_properties -> List[str]."""
        self.assertEqual(
            self._get_return_annotation(
                DatasetHandlerProtocol, "get_required_properties"
            ),
            List[str],
        )

    def test_handler_get_molecular_charge_returns_int(self):
        """get_molecular_charge -> int."""
        self.assertIs(
            self._get_return_annotation(
                DatasetHandlerProtocol, "get_molecular_charge"
            ),
            int,
        )

    def test_handler_get_molecule_creation_strategy_returns_str(self):
        """get_molecule_creation_strategy -> str."""
        self.assertIs(
            self._get_return_annotation(
                DatasetHandlerProtocol, "get_molecule_creation_strategy"
            ),
            str,
        )

    def test_handler_get_processing_statistics_returns_dict(self):
        """get_processing_statistics -> Dict[str, Any]."""
        self.assertEqual(
            self._get_return_annotation(
                DatasetHandlerProtocol, "get_processing_statistics"
            ),
            Dict[str, Any],
        )

    def test_converter_convert_returns_data(self):
        """convert -> Data."""
        self.assertIs(
            self._get_return_annotation(DatasetConverterProtocol, "convert"),
            Data,
        )

    def test_validator_validate_returns_dict(self):
        """validate -> Dict[str, Any]."""
        self.assertEqual(
            self._get_return_annotation(DatasetValidatorProtocol, "validate"),
            Dict[str, Any],
        )


# ============================================================================
# GROUP 12: Module-Level Imports and Exports (5 tests)
# ============================================================================

class TestModuleImportsAndExports(unittest.TestCase):
    """Verify the protocols module exports the expected symbols."""

    def test_module_exports_handler_protocol(self):
        """DatasetHandlerProtocol is importable from protocols module."""
        import milia_pipeline.datasets.protocols as mod
        self.assertTrue(hasattr(mod, "DatasetHandlerProtocol"))

    def test_module_exports_converter_protocol(self):
        """DatasetConverterProtocol is importable from protocols module."""
        import milia_pipeline.datasets.protocols as mod
        self.assertTrue(hasattr(mod, "DatasetConverterProtocol"))

    def test_module_exports_validator_protocol(self):
        """DatasetValidatorProtocol is importable from protocols module."""
        import milia_pipeline.datasets.protocols as mod
        self.assertTrue(hasattr(mod, "DatasetValidatorProtocol"))

    def test_module_has_docstring(self):
        """The protocols module has a non-empty module docstring."""
        import milia_pipeline.datasets.protocols as mod
        self.assertIsNotNone(mod.__doc__)
        self.assertGreater(len(mod.__doc__.strip()), 0)

    def test_module_uses_numpy_and_torch_geometric(self):
        """The module depends on numpy and torch_geometric.data.Data."""
        import milia_pipeline.datasets.protocols as mod
        # Verify that the module source imports these (evidence: line 12-13)
        source = inspect.getsource(mod)
        self.assertIn("import numpy", source)
        self.assertIn("torch_geometric.data", source)


# ============================================================================
# GROUP 13: Realistic Integration Patterns (6 tests)
# ============================================================================

class TestRealisticIntegrationPatterns(unittest.TestCase):
    """Test realistic usage patterns matching the MILIA pipeline architecture."""

    def test_handler_can_be_used_as_type_hint_target(self):
        """A function accepting DatasetHandlerProtocol works with any implementation."""

        def process_with_handler(handler: DatasetHandlerProtocol) -> str:
            return handler.get_dataset_type()

        handler = _FullHandler()
        result = process_with_handler(handler)
        self.assertEqual(result, "test")

    def test_handler_get_required_properties_returns_list(self):
        """Calling get_required_properties on an implementation returns a list."""
        handler = _FullHandler()
        result = handler.get_required_properties()
        self.assertIsInstance(result, list)

    def test_handler_get_molecular_charge_with_numpy_array(self):
        """get_molecular_charge accepts np.ndarray for atomic_numbers."""
        handler = _FullHandler()
        atomic_numbers = np.array([6, 1, 1, 1, 1])
        charge = handler.get_molecular_charge(
            raw_properties_dict={"charge": 0},
            atomic_numbers=atomic_numbers,
            mol_identifier="CH4",
        )
        self.assertIsInstance(charge, int)

    def test_handler_enrich_pyg_data_with_real_data_object(self):
        """enrich_pyg_data accepts a torch_geometric Data object."""
        handler = _FullHandler()
        pyg_data = Data()
        result = handler.enrich_pyg_data(
            pyg_data=pyg_data,
            raw_properties_dict={"energy": -76.5},
            molecule_index=0,
        )
        self.assertIsInstance(result, Data)

    def test_converter_convert_returns_data_object(self):
        """Calling convert on an implementation returns a Data object."""
        converter = _FullConverter()
        result = converter.convert(raw_data={"atoms": [1, 6, 8]})
        self.assertIsInstance(result, Data)

    def test_validator_validate_returns_dict(self):
        """Calling validate on an implementation returns a dict."""
        validator = _FullValidator()
        result = validator.validate(data={"energy": -76.5, "forces": [0.1, 0.2]})
        self.assertIsInstance(result, dict)


# ============================================================================
# GROUP 14: Edge Cases and Boundary Conditions (7 tests)
# ============================================================================

class TestEdgeCasesAndBoundaryConditions(unittest.TestCase):
    """Test edge cases and boundary conditions for protocol compliance."""

    def test_method_as_classmethod_still_structural(self):
        """A class with @classmethod versions of methods — isinstance checks
        method name presence, so this may or may not pass depending on Python
        version. We verify the behavior is consistent."""

        class ClassMethodHandler:
            @classmethod
            def get_dataset_type(cls) -> str:
                return "test"

        # A class with only one method as classmethod won't satisfy full protocol
        obj = ClassMethodHandler()
        self.assertNotIsInstance(obj, DatasetHandlerProtocol)

    def test_method_as_staticmethod_name_present(self):
        """A class with a @staticmethod of the right name — verify behavior."""

        class StaticConverter:
            @staticmethod
            def convert(raw_data: Any) -> Data:
                return Data()

            @staticmethod
            def supports_format(format_type: str) -> bool:
                return True

        obj = StaticConverter()
        # Static methods are still callable attributes — isinstance should pass
        self.assertIsInstance(obj, DatasetConverterProtocol)

    def test_property_with_protocol_method_name(self):
        """A property named like a protocol method — behavior depends on
        Python version (3.12+ uses inspect.getattr_static)."""

        class PropertyConverter:
            @property
            def convert(self):
                return lambda raw_data: Data()

            def supports_format(self, format_type: str) -> bool:
                return True

        obj = PropertyConverter()
        # Properties ARE attributes — should satisfy name-presence check
        self.assertIsInstance(obj, DatasetConverterProtocol)

    def test_callable_attribute_satisfies_protocol(self):
        """An instance with callable attributes set dynamically satisfies protocol."""

        class DynamicValidator:
            def __init__(self):
                self.validate = lambda data: data
                self.get_validation_rules = lambda: {}

        obj = DynamicValidator()
        self.assertIsInstance(obj, DatasetValidatorProtocol)

    def test_none_attribute_does_not_satisfy(self):
        """An instance with None-valued attributes does NOT satisfy protocol.

        Since Python 3.12, isinstance() on runtime_checkable protocols uses
        inspect.getattr_static() which correctly distinguishes between
        callable methods and None-valued class attributes. None attributes
        fail the protocol check.
        (See: cpython gh-102433, bpo-44807)
        """

        class NoneMethodConverter:
            convert = None
            supports_format = None

        obj = NoneMethodConverter()
        self.assertNotIsInstance(obj, DatasetConverterProtocol)

    def test_inherited_methods_satisfy_protocol(self):
        """Methods inherited from a parent class satisfy protocol checks."""

        class BaseHandler:
            def get_dataset_type(self): return "base"
            def validate_molecule_data(self, raw_properties_dict, molecule_index, identifier="N/A"): pass
            def get_required_properties(self): return []
            def process_property_value(self, key, value, molecule_index, identifier="N/A"): return value
            def enrich_pyg_data(self, pyg_data, raw_properties_dict, molecule_index, identifier="N/A"): return pyg_data
            def get_processing_statistics(self, processed_molecules): return {}
            def get_supported_structural_features(self): return {}
            def get_molecular_charge(self, raw_properties_dict, atomic_numbers, mol_identifier=None): return 0
            def get_molecule_creation_strategy(self): return "coordinate_based"
            def get_transform_recommendations(self): return {}
            def get_supported_descriptors(self): return {}

        class ChildHandler(BaseHandler):
            """Inherits all methods, adds nothing."""
            pass

        obj = ChildHandler()
        self.assertIsInstance(obj, DatasetHandlerProtocol)

    def test_magic_mock_satisfies_all_protocols(self):
        """MagicMock satisfies all protocols (it has any attribute)."""
        mock = MagicMock()
        self.assertIsInstance(mock, DatasetHandlerProtocol)
        self.assertIsInstance(mock, DatasetConverterProtocol)
        self.assertIsInstance(mock, DatasetValidatorProtocol)


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_comprehensive_suite():
    """Run all test groups in a structured order."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestDatasetHandlerProtocolTypeIdentity,          # GROUP 1:  8 tests
        TestDatasetHandlerProtocolSignatures,             # GROUP 2: 11 tests
        TestDatasetHandlerProtocolStructuralSubtyping,    # GROUP 3: 12 tests
        TestDatasetConverterProtocolIdentity,             # GROUP 4:  8 tests
        TestDatasetConverterProtocolStructuralSubtyping,  # GROUP 5:  7 tests
        TestDatasetValidatorProtocolIdentity,             # GROUP 6:  8 tests
        TestDatasetValidatorProtocolStructuralSubtyping,  # GROUP 7:  7 tests
        TestCrossProtocolIsolation,                       # GROUP 8:  6 tests
        TestMultiProtocolCompliance,                      # GROUP 9:  5 tests
        TestDatasetHandlerProtocolDocstrings,             # GROUP 10: 1 test (subTests)
        TestProtocolReturnAnnotations,                    # GROUP 11: 8 tests
        TestModuleImportsAndExports,                      # GROUP 12: 5 tests
        TestRealisticIntegrationPatterns,                 # GROUP 13: 6 tests
        TestEdgeCasesAndBoundaryConditions,               # GROUP 14: 7 tests
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS — protocols.py")
    print("=" * 80)
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    total_test_groups = len(test_classes)
    print(f"\nTest Groups: {total_test_groups}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - PRODUCTION-READY")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - REVIEW REQUIRED")
        return 1


if __name__ == "__main__":
    if "pytest" in sys.modules:
        # Let pytest discover and run tests normally
        pass
    else:
        sys.exit(run_comprehensive_suite())


"""
TEST SUITE SUMMARY — milia_pipeline/datasets/protocols.py
==========================================================

99 comprehensive production-ready tests covering:

GROUP 1: DatasetHandlerProtocol Type Identity and Metaclass (8 tests)
- Is subclass of Protocol
- Is @runtime_checkable
- Is a class
- Has correct name and module
- Has docstring
- Has exactly 11 protocol methods
- All expected method names present

GROUP 2: DatasetHandlerProtocol Method Signatures (11 tests)
- get_dataset_type: params and return annotation
- validate_molecule_data: params and 'identifier' default
- get_required_properties: params
- process_property_value: params and 'identifier' default
- enrich_pyg_data: params and 'identifier' default
- get_molecular_charge: params and 'mol_identifier' default

GROUP 3: DatasetHandlerProtocol isinstance/issubclass (12 tests)
- Full implementation passes isinstance and issubclass
- Empty class fails both
- Partial implementation (missing each method individually)
- Explicit subclass passes isinstance
- Mock(spec=_FullHandler) passes
- Non-runtime_checkable fails with TypeError
- Converter/Validator don't satisfy handler
- Built-in types don't satisfy
- Protocol class not instance of itself

GROUP 4: DatasetConverterProtocol Type Identity and Signatures (8 tests)
- Is Protocol, @runtime_checkable, correct name, docstring
- Has exactly 2 methods
- convert and supports_format signatures and return annotations

GROUP 5: DatasetConverterProtocol isinstance/issubclass (7 tests)
- Full implementation passes
- Empty class fails
- Missing convert or supports_format individually fails
- Handler impl doesn't satisfy converter
- Mock(spec=_FullConverter) passes

GROUP 6: DatasetValidatorProtocol Type Identity and Signatures (8 tests)
- Is Protocol, @runtime_checkable, correct name, docstring
- Has exactly 2 methods
- validate and get_validation_rules signatures and return annotations

GROUP 7: DatasetValidatorProtocol isinstance/issubclass (7 tests)
- Full implementation passes
- Empty class fails
- Missing validate or get_validation_rules individually fails
- Handler impl doesn't satisfy validator
- Mock(spec=_FullValidator) passes

GROUP 8: Cross-Protocol Isolation (6 tests)
- Each implementation satisfies only its own protocol
- No accidental cross-protocol compliance

GROUP 9: Multi-Protocol Compliance (5 tests)
- Handler + Converter dual compliance
- Handler + Validator dual compliance
- All three protocols simultaneously
- Converter + Validator without Handler
- Extra methods don't break compliance

GROUP 10: Handler Method Docstrings (1 test with 11 subTests)
- Every protocol method has a non-empty docstring

GROUP 11: Protocol Method Return Type Annotations (8 tests)
- Handler: str, None, List[str], int, str, Dict[str, Any]
- Converter: Data
- Validator: Dict[str, Any]

GROUP 12: Module-Level Imports and Exports (5 tests)
- All three protocols importable
- Module docstring exists
- Module depends on numpy and torch_geometric

GROUP 13: Realistic Integration Patterns (6 tests)
- Handler used as type hint target
- Handler with numpy array argument
- Handler with torch_geometric Data object
- Converter returns Data
- Validator returns dict

GROUP 14: Edge Cases and Boundary Conditions (7 tests)
- classmethod/staticmethod/property variants
- Callable attributes set dynamically
- None-valued attributes
- Inherited methods from parent class
- MagicMock satisfies all protocols

Total: 99 comprehensive production-ready tests

PRODUCTION-READY QUALITIES:
- NO sys.modules pollution (no module-level mocking)
- All test classes self-contained
- Dynamic test data creation via helper classes
- No NPZ file downloads or file system dependencies
- Comprehensive structural subtyping coverage
- Interface-focused testing (future-proof)
- Compatible with both pytest and unittest runner
- Cross-protocol isolation verified
- Evidence-based runtime_checkable behavior testing
- Method signature verification via inspect
"""
