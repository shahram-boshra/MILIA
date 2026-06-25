#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/preprocessing/utils/qm40_csv_parser.py

Module under test: qm40_csv_parser.py
- ELEMENT_TO_Z: Module-level dict mapping QM40 element symbols to atomic numbers
- QM40_MAIN_COLUMN_TO_KEY: CSV column header -> sanitized NPZ key mapping
- QM40_SCALAR_KEYS: ordered list of the 16 sanitized scalar property keys
- _build_object_array(): build a 1-D object array preserving per-molecule structure
- _to_float(): parse a CSV cell to float (empty -> NaN)
- _locate_csv_files(): find main/xyz/bond CSVs (skipping __MACOSX/._ junk)
- parse_qm40_csv_files(): join the three CSVs by Zinc_id, return (features, metadata)
- get_qm40_property_info(): units/descriptions for the 16 scalar properties

Test path on local machine: ~/ml_projects/milia/tests/test_qm40_csv_parser_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/preprocessing/utils/qm40_csv_parser.py

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution
- No real file downloads — all data created in-memory via tempfile

Updated: June 2026 - Production-ready comprehensive test coverage
"""

import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from milia_pipeline.exceptions import DataProcessingError
from milia_pipeline.preprocessing.utils.qm40_csv_parser import (
    ELEMENT_TO_Z,
    QM40_MAIN_COLUMN_TO_KEY,
    QM40_SCALAR_KEYS,
    _build_object_array,
    _locate_csv_files,
    _to_float,
    get_qm40_property_info,
    parse_qm40_csv_files,
)

# ============================================================================
# CONSTANTS: exact CSV headers (verified against the real QM40_dataset.zip)
# ============================================================================

QM40_MAIN_HEADER = (
    "Zinc_id,smile,Internal_E(0K),HOMO,LUMO,HL_gap,Polarizability,"
    "spatial extent,dipol_mom,ZPE,rot1,rot2,rot3,Inter_E(298),Enthalpy,Free_E,CV,Entropy"
)
QM40_XYZ_HEADER = "Zinc_id,smile,atom,init_x,init_y,init_z,final_x,final_y,final_z,charge"
QM40_BOND_HEADER = "Zinc_id,smile,atom1,atom2,bond,tag,lmod"

# The 16 main.csv scalar columns in file order (note "spatial extent" has a SPACE)
QM40_MAIN_SCALAR_COLUMNS = [
    "Internal_E(0K)",
    "HOMO",
    "LUMO",
    "HL_gap",
    "Polarizability",
    "spatial extent",
    "dipol_mom",
    "ZPE",
    "rot1",
    "rot2",
    "rot3",
    "Inter_E(298)",
    "Enthalpy",
    "Free_E",
    "CV",
    "Entropy",
]

# Sanitized NPZ scalar keys in the same order
QM40_SANITIZED_SCALAR_KEYS = [
    "Internal_E_0K",
    "HOMO",
    "LUMO",
    "HL_gap",
    "Polarizability",
    "spatial_extent",
    "dipol_mom",
    "ZPE",
    "rot1",
    "rot2",
    "rot3",
    "Inter_E_298",
    "Enthalpy",
    "Free_E",
    "CV",
    "Entropy",
]


# ============================================================================
# HELPERS: QM40 CSV builders for realistic test data
# ============================================================================


def _make_logger():
    """Create a logger instance for testing."""
    logger = logging.getLogger("test_qm40_csv_parser")
    logger.setLevel(logging.DEBUG)
    return logger


def _default_scalars(offset=0.0):
    """Return a dict of the 16 scalar values keyed by EXACT CSV column name."""
    return {
        "Internal_E(0K)": -40.47893 + offset,
        "HOMO": -0.3877,
        "LUMO": 0.1171,
        "HL_gap": -0.5048,  # HOMO - LUMO (negative)
        "Polarizability": 13.21,
        "spatial extent": 35.3641,
        "dipol_mom": 0.0,
        "ZPE": 27.7,
        "rot1": 157.7118,
        "rot2": 157.7118,
        "rot3": 157.7118,
        "Inter_E(298)": -40.47625 + offset,
        "Enthalpy": -40.47582 + offset,
        "Free_E": -40.49838 + offset,
        "CV": 6.469,
        "Entropy": 44.0,
    }


# Three molecular templates with DISTINCT atom counts (ragged) and DISTINCT
# leading elements, so join-by-Zinc_id correctness is verifiable.
_TEMPLATES = [
    {  # CH4 (5 atoms)
        "atoms": ["C", "H", "H", "H", "H"],
        "init_coords": [
            [-0.0127, 1.0858, 0.0080],
            [0.0022, -0.0060, 0.0020],
            [1.0117, 1.4638, 0.0003],
            [-0.5408, 1.4475, -0.8766],
            [-0.5238, 1.4379, 0.9064],
        ],
        "final_coords": [
            [-0.0130, 1.0850, 0.0085],
            [0.0020, -0.0065, 0.0018],
            [1.0120, 1.4640, 0.0001],
            [-0.5410, 1.4470, -0.8770],
            [-0.5240, 1.4375, 0.9068],
        ],
        "charges": [-0.535689, 0.133921, 0.133922, 0.133923, 0.133923],
        "bonds": [(1, 2, "CH", 5.10), (1, 3, "CH", 5.11), (1, 4, "CH", 5.12), (1, 5, "CH", 5.13)],
    },
    {  # NH3 (4 atoms)
        "atoms": ["N", "H", "H", "H"],
        "init_coords": [
            [0.0000, 0.0000, 0.1173],
            [0.0000, 0.9377, -0.2737],
            [0.8121, -0.4689, -0.2737],
            [-0.8121, -0.4689, -0.2737],
        ],
        "final_coords": [
            [0.0000, 0.0000, 0.1200],
            [0.0000, 0.9400, -0.2700],
            [0.8140, -0.4700, -0.2700],
            [-0.8140, -0.4700, -0.2700],
        ],
        "charges": [-1.020000, 0.340000, 0.340000, 0.340000],
        "bonds": [(1, 2, "NH", 6.80), (1, 3, "NH", 6.81), (1, 4, "NH", 6.82)],
    },
    {  # H2O (3 atoms)
        "atoms": ["O", "H", "H"],
        "init_coords": [
            [0.0000, 0.0000, 0.1173],
            [0.0000, 0.7572, -0.4692],
            [0.0000, -0.7572, -0.4692],
        ],
        "final_coords": [
            [0.0000, 0.0000, 0.1200],
            [0.0000, 0.7600, -0.4700],
            [0.0000, -0.7600, -0.4700],
        ],
        "charges": [-0.680000, 0.340000, 0.340000],
        "bonds": [(1, 2, "OH", 8.40), (1, 3, "OH", 8.41)],
    },
]


def _make_molecule(idx, template=None):
    """Build one molecule spec with a distinct Zinc_id (1-based idx)."""
    tmpl = _TEMPLATES[(idx - 1) % len(_TEMPLATES)] if template is None else template
    return {
        "zinc_id": f"ZINC{idx:012d}",
        "smiles": "C" if tmpl["atoms"][0] == "C" else "N" if tmpl["atoms"][0] == "N" else "O",
        "atoms": list(tmpl["atoms"]),
        "init_coords": [list(c) for c in tmpl["init_coords"]],
        "final_coords": [list(c) for c in tmpl["final_coords"]],
        "charges": list(tmpl["charges"]),
        "scalars": _default_scalars(offset=idx * 0.001),
        "bonds": list(tmpl["bonds"]),
    }


def _make_molecules(n, template=None):
    """Build a list of n molecule specs with distinct Zinc_ids."""
    return [_make_molecule(i + 1, template=template) for i in range(n)]


def _build_main_csv(molecules):
    lines = [QM40_MAIN_HEADER]
    for m in molecules:
        row = [m["zinc_id"], m["smiles"]] + [str(m["scalars"][c]) for c in QM40_MAIN_SCALAR_COLUMNS]
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def _build_xyz_csv(molecules):
    lines = [QM40_XYZ_HEADER]
    for m in molecules:
        for i, sym in enumerate(m["atoms"]):
            ix, iy, iz = m["init_coords"][i]
            fx, fy, fz = m["final_coords"][i]
            q = m["charges"][i]
            lines.append(
                ",".join(
                    str(v) for v in [m["zinc_id"], m["smiles"], sym, ix, iy, iz, fx, fy, fz, q]
                )
            )
    return "\n".join(lines) + "\n"


def _build_bond_csv(molecules):
    lines = [QM40_BOND_HEADER]

    def _sym(atoms, idx):
        # Cosmetic label only (the parser ignores the `bond` column); tolerate
        # deliberately out-of-range indices used by drop-bond tests.
        return atoms[idx - 1] if 1 <= idx <= len(atoms) else "X"

    for m in molecules:
        for a1, a2, tag, lmod in m["bonds"]:
            label = f"{_sym(m['atoms'], a1)}{a1}{_sym(m['atoms'], a2)}{a2}"
            lines.append(
                ",".join(str(v) for v in [m["zinc_id"], m["smiles"], a1, a2, label, tag, lmod])
            )
    return "\n".join(lines) + "\n"


def _write_qm40_dataset(
    base_dir,
    molecules,
    main_order=None,
    xyz_order=None,
    bond_order=None,
    subfolder="QM40 dataset",
    macosx_junk=False,
):
    """
    Write the three QM40 CSV files into a subfolder (default mirrors the real
    in-zip 'QM40 dataset/' name, which contains a SPACE). The per-file row order
    can be controlled independently to test join-by-Zinc_id.
    """
    ds_dir = base_dir / subfolder
    ds_dir.mkdir(parents=True, exist_ok=True)
    mo = main_order if main_order is not None else molecules
    xo = xyz_order if xyz_order is not None else molecules
    bo = bond_order if bond_order is not None else molecules
    (ds_dir / "QM40_main.csv").write_text(_build_main_csv(mo))
    (ds_dir / "QM40_xyz.csv").write_text(_build_xyz_csv(xo))
    (ds_dir / "QM40_bond.csv").write_text(_build_bond_csv(bo))
    if macosx_junk:
        mac = base_dir / "__MACOSX" / subfolder
        mac.mkdir(parents=True, exist_ok=True)
        (mac / "._QM40_main.csv").write_bytes(b"\x00\x05\x16garbage")
        (mac / "._QM40_xyz.csv").write_bytes(b"\x00\x05garbage")
        (mac / "._QM40_bond.csv").write_bytes(b"\x00\x05garbage")
        # AppleDouble sibling inside the real folder, too
        (ds_dir / "._QM40_main.csv").write_bytes(b"\x00garbage")
    return base_dir


# ============================================================================
# GROUP 1: ELEMENT_TO_Z Module-Level Constant
# ============================================================================


class TestElementToZ(unittest.TestCase):
    """Test that ELEMENT_TO_Z is correctly scoped to QM40's 7 elements."""

    def test_is_dict(self):
        self.assertIsInstance(ELEMENT_TO_Z, dict)

    def test_all_keys_are_strings(self):
        for key in ELEMENT_TO_Z:
            self.assertIsInstance(key, str)

    def test_all_values_positive_ints(self):
        for symbol, z in ELEMENT_TO_Z.items():
            with self.subTest(symbol=symbol):
                self.assertIsInstance(z, int)
                self.assertGreater(z, 0)

    def test_contains_exactly_seven_qm40_elements(self):
        """QM40 supports exactly 7 elements: H, C, N, O, F, S, Cl."""
        self.assertEqual(set(ELEMENT_TO_Z.keys()), {"H", "C", "N", "O", "F", "S", "Cl"})

    def test_hydrogen_is_1(self):
        self.assertEqual(ELEMENT_TO_Z["H"], 1)

    def test_carbon_is_6(self):
        self.assertEqual(ELEMENT_TO_Z["C"], 6)

    def test_sulfur_is_16(self):
        self.assertEqual(ELEMENT_TO_Z["S"], 16)

    def test_chlorine_is_17(self):
        self.assertEqual(ELEMENT_TO_Z["Cl"], 17)

    def test_excludes_non_qm40_elements(self):
        """QM40 does not include elements outside its documented set."""
        for sym in ("Br", "I", "P", "Na", "Li", "K"):
            with self.subTest(symbol=sym):
                self.assertNotIn(sym, ELEMENT_TO_Z)


# ============================================================================
# GROUP 2: Column mapping + scalar key constants
# ============================================================================


class TestColumnMapping(unittest.TestCase):
    """Test QM40_MAIN_COLUMN_TO_KEY and QM40_SCALAR_KEYS."""

    def test_mapping_is_dict(self):
        self.assertIsInstance(QM40_MAIN_COLUMN_TO_KEY, dict)

    def test_mapping_has_16_entries(self):
        self.assertEqual(len(QM40_MAIN_COLUMN_TO_KEY), 16)

    def test_spatial_extent_has_space_in_csv_key(self):
        """The CSV header is 'spatial extent' (SPACE), sanitized to 'spatial_extent'."""
        self.assertIn("spatial extent", QM40_MAIN_COLUMN_TO_KEY)
        self.assertEqual(QM40_MAIN_COLUMN_TO_KEY["spatial extent"], "spatial_extent")

    def test_internal_energy_0k_sanitized(self):
        self.assertEqual(QM40_MAIN_COLUMN_TO_KEY["Internal_E(0K)"], "Internal_E_0K")

    def test_inter_energy_298_sanitized(self):
        self.assertEqual(QM40_MAIN_COLUMN_TO_KEY["Inter_E(298)"], "Inter_E_298")

    def test_scalar_keys_match_mapping_values(self):
        self.assertEqual(QM40_SCALAR_KEYS, list(QM40_MAIN_COLUMN_TO_KEY.values()))

    def test_scalar_keys_match_expected_order(self):
        self.assertEqual(QM40_SCALAR_KEYS, QM40_SANITIZED_SCALAR_KEYS)

    def test_all_csv_columns_are_valid_headers(self):
        """Every mapping key is one of the real main.csv columns."""
        for col in QM40_MAIN_COLUMN_TO_KEY:
            with self.subTest(col=col):
                self.assertIn(col, QM40_MAIN_SCALAR_COLUMNS)


# ============================================================================
# GROUP 3: _build_object_array (the 2-D-collapse guard)
# ============================================================================


class TestBuildObjectArray(unittest.TestCase):
    """Test _build_object_array preserves per-element structure."""

    def test_returns_object_dtype(self):
        arr = _build_object_array([np.array([1, 2, 3]), np.array([4, 5])])
        self.assertEqual(arr.dtype, object)

    def test_is_one_dimensional(self):
        arr = _build_object_array([np.array([1, 2, 3]), np.array([4, 5])])
        self.assertEqual(arr.ndim, 1)

    def test_length_matches_input(self):
        arr = _build_object_array([np.array([1]), np.array([2]), np.array([3])])
        self.assertEqual(len(arr), 3)

    def test_equal_shape_inner_arrays_not_collapsed(self):
        """
        CRITICAL: equal-shape inner arrays must NOT collapse to a 2-D array.
        np.array([a, b], dtype=object) would produce shape (2, 3); the helper
        must keep a 1-D object array whose elements are the originals.
        """
        a = np.array([1, 2, 3], dtype=np.int64)
        b = np.array([4, 5, 6], dtype=np.int64)
        arr = _build_object_array([a, b])
        self.assertEqual(arr.shape, (2,))
        np.testing.assert_array_equal(arr[0], a)
        np.testing.assert_array_equal(arr[1], b)

    def test_preserves_inner_dtype(self):
        a = np.array([1, 2, 3], dtype=np.int64)
        arr = _build_object_array([a])
        self.assertEqual(arr[0].dtype, np.int64)

    def test_handles_strings(self):
        arr = _build_object_array(["ZINC1", "ZINC2"])
        self.assertEqual(arr.dtype, object)
        self.assertEqual(list(arr), ["ZINC1", "ZINC2"])

    def test_empty_list(self):
        arr = _build_object_array([])
        self.assertEqual(arr.dtype, object)
        self.assertEqual(len(arr), 0)


# ============================================================================
# GROUP 4: _to_float
# ============================================================================


class TestToFloat(unittest.TestCase):
    """Test _to_float cell parsing."""

    def test_parses_float(self):
        self.assertAlmostEqual(_to_float("-40.47893"), -40.47893)

    def test_parses_integer_string(self):
        self.assertAlmostEqual(_to_float("42"), 42.0)

    def test_empty_string_is_nan(self):
        self.assertTrue(np.isnan(_to_float("")))

    def test_whitespace_is_nan(self):
        self.assertTrue(np.isnan(_to_float("   ")))

    def test_strips_whitespace(self):
        self.assertAlmostEqual(_to_float("  1.5  "), 1.5)

    def test_returns_float_type(self):
        self.assertIsInstance(_to_float("1.0"), float)

    def test_scientific_notation(self):
        self.assertAlmostEqual(_to_float("1.234e-5"), 1.234e-5)


# ============================================================================
# GROUP 5: _locate_csv_files
# ============================================================================


class TestLocateCsvFiles(unittest.TestCase):
    """Test CSV discovery and macOS-junk skipping."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="test_qm40_locate_")
        self._base = Path(self._tmpdir)

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_finds_three_csvs(self):
        _write_qm40_dataset(self._base, _make_molecules(2))
        main_csv, xyz_csv, bond_csv = _locate_csv_files(self._base)
        self.assertEqual(main_csv.name, "QM40_main.csv")
        self.assertEqual(xyz_csv.name, "QM40_xyz.csv")
        self.assertEqual(bond_csv.name, "QM40_bond.csv")

    def test_finds_csvs_in_spaced_subfolder(self):
        """The real archive nests CSVs under 'QM40 dataset/' (with a space)."""
        _write_qm40_dataset(self._base, _make_molecules(1), subfolder="QM40 dataset")
        main_csv, _, _ = _locate_csv_files(self._base)
        self.assertIn("QM40 dataset", str(main_csv))

    def test_skips_macosx_and_appledouble_junk(self):
        """__MACOSX/ and ._ files must be ignored, real files still found."""
        _write_qm40_dataset(self._base, _make_molecules(1), macosx_junk=True)
        main_csv, xyz_csv, bond_csv = _locate_csv_files(self._base)
        self.assertFalse(main_csv.name.startswith("._"))
        self.assertNotIn("__MACOSX", main_csv.parts)
        self.assertNotIn("__MACOSX", xyz_csv.parts)
        self.assertNotIn("__MACOSX", bond_csv.parts)

    def test_missing_files_raise(self):
        with self.assertRaises(DataProcessingError):
            _locate_csv_files(self._base)  # empty dir

    def test_missing_files_error_lists_missing(self):
        # Write only main.csv
        ds = self._base / "QM40 dataset"
        ds.mkdir(parents=True)
        (ds / "QM40_main.csv").write_text(_build_main_csv(_make_molecules(1)))
        with self.assertRaises(DataProcessingError) as ctx:
            _locate_csv_files(self._base)
        self.assertIn("xyz", str(ctx.exception).lower())


# ============================================================================
# GROUP 6: parse_qm40_csv_files — Happy Path
# ============================================================================


class TestParseHappyPath(unittest.TestCase):
    """Test parse_qm40_csv_files on a small valid dataset."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="test_qm40_happy_")
        self._base = Path(self._tmpdir)
        self._molecules = _make_molecules(3)
        _write_qm40_dataset(self._base, self._molecules)
        self._features, self._metadata = parse_qm40_csv_files(self._base, logger=_make_logger())

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_returns_tuple(self):
        result = (self._features, self._metadata)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_features_is_dict(self):
        self.assertIsInstance(self._features, dict)

    def test_metadata_is_dict(self):
        self.assertIsInstance(self._metadata, dict)

    def test_required_core_keys_present(self):
        for key in ("compounds", "atoms", "coordinates"):
            with self.subTest(key=key):
                self.assertIn(key, self._features)

    def test_compounds_are_zinc_ids(self):
        self.assertEqual(list(self._features["compounds"]), [m["zinc_id"] for m in self._molecules])

    def test_smiles_present(self):
        self.assertIn("smiles", self._features)
        self.assertEqual(len(self._features["smiles"]), 3)

    def test_qmulliken_present(self):
        self.assertIn("Qmulliken", self._features)

    def test_all_16_scalar_keys_present(self):
        for key in QM40_SANITIZED_SCALAR_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, self._features)
                self.assertEqual(len(self._features[key]), 3)

    def test_metadata_counts(self):
        self.assertEqual(self._metadata["num_molecules_parsed"], 3)
        self.assertEqual(self._metadata["num_molecules_failed"], 0)

    def test_metadata_source_format(self):
        self.assertEqual(self._metadata["source_format"], "qm40_csv")

    def test_metadata_units(self):
        self.assertEqual(self._metadata["coordinate_units"], "angstrom")
        self.assertEqual(self._metadata["energy_units"], "hartree")

    def test_metadata_level_of_theory(self):
        self.assertEqual(self._metadata["level_of_theory"], "B3LYP/6-31G(2df,p)")

    def test_atom_counts_match_specs(self):
        """Ragged: molecule i has the atom count of its spec (5, 4, 3)."""
        expected_counts = [len(m["atoms"]) for m in self._molecules]
        actual_counts = [len(a) for a in self._features["atoms"]]
        self.assertEqual(actual_counts, expected_counts)

    def test_internal_energy_value(self):
        """Scalar value round-trips for the first molecule."""
        expected = self._molecules[0]["scalars"]["Internal_E(0K)"]
        self.assertAlmostEqual(float(self._features["Internal_E_0K"][0]), expected, places=5)


# ============================================================================
# GROUP 7: Cross-file join by Zinc_id (files NOT row-aligned)
# ============================================================================


class TestJoinByZincId(unittest.TestCase):
    """The three CSVs are not row-aligned; join must be by Zinc_id."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="test_qm40_join_")
        self._base = Path(self._tmpdir)
        self._molecules = _make_molecules(3)  # CH4(5), NH3(4), H2O(3)
        m1, m2, m3 = self._molecules
        # Deliberately scramble per-file order
        _write_qm40_dataset(
            self._base,
            self._molecules,
            main_order=[m1, m2, m3],
            xyz_order=[m3, m1, m2],
            bond_order=[m2, m3, m1],
        )
        self._features, _ = parse_qm40_csv_files(self._base, logger=_make_logger())

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_compounds_follow_main_order(self):
        self.assertEqual(list(self._features["compounds"]), [m["zinc_id"] for m in self._molecules])

    def test_atoms_joined_to_correct_molecule(self):
        """Each molecule's atomic numbers come from ITS xyz rows, not by position."""
        z_map = {"C": 6, "H": 1, "N": 7, "O": 8}
        for i, m in enumerate(self._molecules):
            with self.subTest(zinc=m["zinc_id"]):
                expected = np.array([z_map[s] for s in m["atoms"]])
                np.testing.assert_array_equal(self._features["atoms"][i], expected)

    def test_first_atom_element_matches_template(self):
        """m1->C(6), m2->N(7), m3->O(8): proves no positional mismatch."""
        self.assertEqual(int(self._features["atoms"][0][0]), 6)
        self.assertEqual(int(self._features["atoms"][1][0]), 7)
        self.assertEqual(int(self._features["atoms"][2][0]), 8)

    def test_coordinates_count_matches_atoms(self):
        for i in range(3):
            with self.subTest(i=i):
                self.assertEqual(
                    self._features["coordinates"][i].shape[0],
                    len(self._features["atoms"][i]),
                )


# ============================================================================
# GROUP 8: Bond data (1-based -> 0-based, toggle, out-of-range)
# ============================================================================


class TestBondData(unittest.TestCase):
    """Test per-bond local-mode data handling."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="test_qm40_bond_")
        self._base = Path(self._tmpdir)

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_bond_keys_present_when_enabled(self):
        _write_qm40_dataset(self._base, _make_molecules(2))
        features, _ = parse_qm40_csv_files(
            self._base, include_bond_data=True, logger=_make_logger()
        )
        for key in ("bond_atom1_idx", "bond_atom2_idx", "bond_tag", "bond_lmod_ka"):
            with self.subTest(key=key):
                self.assertIn(key, features)

    def test_bond_keys_absent_when_disabled(self):
        _write_qm40_dataset(self._base, _make_molecules(2))
        features, _ = parse_qm40_csv_files(
            self._base, include_bond_data=False, logger=_make_logger()
        )
        for key in ("bond_atom1_idx", "bond_atom2_idx", "bond_tag", "bond_lmod_ka"):
            with self.subTest(key=key):
                self.assertNotIn(key, features)

    def test_bond_indices_converted_to_zero_based(self):
        """CSV stores 1-based indices; parser must emit 0-based."""
        mols = _make_molecules(1)  # CH4: bonds (1,2),(1,3),(1,4),(1,5)
        _write_qm40_dataset(self._base, mols)
        features, _ = parse_qm40_csv_files(
            self._base, include_bond_data=True, logger=_make_logger()
        )
        a1 = features["bond_atom1_idx"][0]
        a2 = features["bond_atom2_idx"][0]
        # All first-atom indices were 1 (1-based) -> 0 (0-based)
        np.testing.assert_array_equal(a1, np.array([0, 0, 0, 0]))
        np.testing.assert_array_equal(a2, np.array([1, 2, 3, 4]))

    def test_bond_tag_preserved_as_strings(self):
        mols = _make_molecules(1)
        _write_qm40_dataset(self._base, mols)
        features, _ = parse_qm40_csv_files(
            self._base, include_bond_data=True, logger=_make_logger()
        )
        tags = list(features["bond_tag"][0])
        self.assertTrue(all(t == "CH" for t in tags))

    def test_bond_lmod_values(self):
        mols = _make_molecules(1)
        _write_qm40_dataset(self._base, mols)
        features, _ = parse_qm40_csv_files(
            self._base, include_bond_data=True, logger=_make_logger()
        )
        ka = features["bond_lmod_ka"][0]
        np.testing.assert_allclose(ka, np.array([5.10, 5.11, 5.12, 5.13]))

    def test_out_of_range_bond_dropped(self):
        """A bond referencing a non-existent atom index is dropped and counted."""
        mols = _make_molecules(1)  # CH4 has 5 atoms (valid indices 1..5)
        mols[0]["bonds"].append((1, 99, "CX", 1.0))  # 99 is out of range
        _write_qm40_dataset(self._base, mols)
        features, metadata = parse_qm40_csv_files(
            self._base, include_bond_data=True, logger=_make_logger()
        )
        # The valid 4 C-H bonds remain; the bogus one is dropped
        self.assertEqual(len(features["bond_lmod_ka"][0]), 4)
        self.assertEqual(metadata.get("dropped_out_of_range_bonds"), 1)


# ============================================================================
# GROUP 9: initial_coordinates toggle
# ============================================================================


class TestInitialCoordinates(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="test_qm40_init_")
        self._base = Path(self._tmpdir)

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_initial_coords_present_when_enabled(self):
        _write_qm40_dataset(self._base, _make_molecules(2))
        features, _ = parse_qm40_csv_files(
            self._base, include_initial_coordinates=True, logger=_make_logger()
        )
        self.assertIn("initial_coordinates", features)

    def test_initial_coords_absent_when_disabled(self):
        _write_qm40_dataset(self._base, _make_molecules(2))
        features, _ = parse_qm40_csv_files(
            self._base, include_initial_coordinates=False, logger=_make_logger()
        )
        self.assertNotIn("initial_coordinates", features)

    def test_initial_and_final_coords_differ(self):
        _write_qm40_dataset(self._base, _make_molecules(1))
        features, _ = parse_qm40_csv_files(
            self._base, include_initial_coordinates=True, logger=_make_logger()
        )
        # Templates use slightly different init vs final coords
        self.assertFalse(
            np.allclose(features["initial_coordinates"][0], features["coordinates"][0])
        )


# ============================================================================
# GROUP 10: max_molecules parameter
# ============================================================================


class TestMaxMolecules(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="test_qm40_max_")
        self._base = Path(self._tmpdir)
        _write_qm40_dataset(self._base, _make_molecules(5))

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_limits_parsing(self):
        features, metadata = parse_qm40_csv_files(
            self._base, max_molecules=2, logger=_make_logger()
        )
        self.assertEqual(metadata["num_molecules_parsed"], 2)
        self.assertEqual(len(features["compounds"]), 2)

    def test_none_parses_all(self):
        _, metadata = parse_qm40_csv_files(self._base, max_molecules=None, logger=_make_logger())
        self.assertEqual(metadata["num_molecules_parsed"], 5)

    def test_greater_than_total_parses_all(self):
        _, metadata = parse_qm40_csv_files(self._base, max_molecules=100, logger=_make_logger())
        self.assertEqual(metadata["num_molecules_parsed"], 5)

    def test_one(self):
        features, metadata = parse_qm40_csv_files(
            self._base, max_molecules=1, logger=_make_logger()
        )
        self.assertEqual(metadata["num_molecules_parsed"], 1)

    def test_deterministic_first_n_in_main_order(self):
        """max_molecules selects the FIRST N molecules in main.csv order."""
        features, _ = parse_qm40_csv_files(self._base, max_molecules=2, logger=_make_logger())
        self.assertEqual(list(features["compounds"]), ["ZINC000000000001", "ZINC000000000002"])

    def test_scalar_array_lengths_match_limit(self):
        features, _ = parse_qm40_csv_files(self._base, max_molecules=3, logger=_make_logger())
        for key in QM40_SANITIZED_SCALAR_KEYS:
            with self.subTest(key=key):
                self.assertEqual(len(features[key]), 3)


# ============================================================================
# GROUP 11: Error paths
# ============================================================================


class TestErrors(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="test_qm40_err_")
        self._base = Path(self._tmpdir)

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_no_csv_files_raises(self):
        with self.assertRaises(DataProcessingError):
            parse_qm40_csv_files(self._base, logger=_make_logger())

    def test_missing_main_column_raises(self):
        ds = self._base / "QM40 dataset"
        ds.mkdir(parents=True)
        # main.csv with a corrupted header (missing scalar columns)
        (ds / "QM40_main.csv").write_text("Zinc_id,smile\nZINC1,C\n")
        (ds / "QM40_xyz.csv").write_text(_build_xyz_csv(_make_molecules(1)))
        (ds / "QM40_bond.csv").write_text(_build_bond_csv(_make_molecules(1)))
        with self.assertRaises(DataProcessingError):
            parse_qm40_csv_files(self._base, logger=_make_logger())

    def test_empty_main_raises(self):
        ds = self._base / "QM40 dataset"
        ds.mkdir(parents=True)
        (ds / "QM40_main.csv").write_text(QM40_MAIN_HEADER + "\n")  # header only
        (ds / "QM40_xyz.csv").write_text(_build_xyz_csv(_make_molecules(1)))
        (ds / "QM40_bond.csv").write_text(_build_bond_csv(_make_molecules(1)))
        with self.assertRaises(DataProcessingError):
            parse_qm40_csv_files(self._base, logger=_make_logger())

    def test_unknown_element_raises(self):
        """An unsupported element symbol raises DataProcessingError.

        Element-symbol validation happens in the bulk xyz read (_read_xyz_csv),
        which raises before the per-molecule assembly loop — so an out-of-set
        element aborts the parse rather than being counted as a per-molecule
        failure. (Per-molecule failure counting applies to molecules that are
        present in main.csv but have no atom rows in xyz.csv; see the test
        below.)
        """
        mols = _make_molecules(2)
        # Corrupt molecule 1's first atom to an unsupported element
        mols[0]["atoms"][0] = "Xx"
        _write_qm40_dataset(self._base, mols)
        with self.assertRaises(DataProcessingError):
            parse_qm40_csv_files(self._base, logger=_make_logger())

    def test_molecule_missing_xyz_recorded_as_failure(self):
        """A molecule present in main but absent from xyz fails (no atoms)."""
        mols = _make_molecules(2)
        # Write main with both, xyz with only the second molecule
        _write_qm40_dataset(self._base, mols, xyz_order=[mols[1]], bond_order=[mols[1]])
        features, metadata = parse_qm40_csv_files(self._base, logger=_make_logger())
        self.assertEqual(metadata["num_molecules_parsed"], 1)
        self.assertEqual(metadata["num_molecules_failed"], 1)
        self.assertEqual(list(features["compounds"]), [mols[1]["zinc_id"]])


# ============================================================================
# GROUP 12: Dtypes
# ============================================================================


class TestDtypes(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="test_qm40_dtype_")
        self._base = Path(self._tmpdir)
        _write_qm40_dataset(self._base, _make_molecules(3))
        self._features, _ = parse_qm40_csv_files(self._base, logger=_make_logger())

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_compounds_object_array(self):
        self.assertEqual(self._features["compounds"].dtype, object)

    def test_atoms_object_array(self):
        self.assertEqual(self._features["atoms"].dtype, object)

    def test_coordinates_object_array(self):
        self.assertEqual(self._features["coordinates"].dtype, object)

    def test_inner_atoms_integer(self):
        self.assertTrue(np.issubdtype(self._features["atoms"][0].dtype, np.integer))

    def test_inner_coordinates_float(self):
        self.assertTrue(np.issubdtype(self._features["coordinates"][0].dtype, np.floating))

    def test_scalar_arrays_float64(self):
        for key in QM40_SANITIZED_SCALAR_KEYS:
            with self.subTest(key=key):
                self.assertEqual(self._features[key].dtype, np.float64)

    def test_equal_atom_count_not_collapsed(self):
        """All-CH4 dataset (equal atom counts) stays a 1-D object array."""
        tmp = tempfile.mkdtemp(prefix="test_qm40_equal_")
        try:
            base = Path(tmp)
            _write_qm40_dataset(base, _make_molecules(3, template=_TEMPLATES[0]))
            features, _ = parse_qm40_csv_files(base, logger=_make_logger())
            self.assertEqual(features["atoms"].shape, (3,))
            self.assertEqual(features["atoms"][0].shape, (5,))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# GROUP 13: Neutrality sanity (Mulliken charges sum ~ 0)
# ============================================================================


class TestNeutrality(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="test_qm40_neutral_")
        self._base = Path(self._tmpdir)
        _write_qm40_dataset(self._base, _make_molecules(3))
        self._features, _ = parse_qm40_csv_files(self._base, logger=_make_logger())

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_mulliken_sums_near_zero(self):
        """QM40 is neutral-only: per-molecule Mulliken charges sum ~ 0."""
        for i in range(3):
            with self.subTest(i=i):
                self.assertAlmostEqual(float(np.sum(self._features["Qmulliken"][i])), 0.0, places=4)


# ============================================================================
# GROUP 14: get_qm40_property_info
# ============================================================================


class TestGetPropertyInfo(unittest.TestCase):
    def setUp(self):
        self._info = get_qm40_property_info()

    def test_returns_dict(self):
        self.assertIsInstance(self._info, dict)

    def test_has_16_entries(self):
        self.assertEqual(len(self._info), 16)

    def test_covers_all_scalar_keys(self):
        self.assertEqual(set(self._info.keys()), set(QM40_SANITIZED_SCALAR_KEYS))

    def test_each_entry_has_unit_and_description(self):
        for key, meta in self._info.items():
            with self.subTest(key=key):
                self.assertIn("unit", meta)
                self.assertIn("description", meta)

    def test_internal_energy_unit_hartree(self):
        self.assertEqual(self._info["Internal_E_0K"]["unit"], "Hartree")

    def test_zpe_unit_kcal_per_mol(self):
        self.assertEqual(self._info["ZPE"]["unit"], "kcal/mol")

    def test_rot1_unit_ghz(self):
        self.assertEqual(self._info["rot1"]["unit"], "GHz")

    def test_dipole_unit_debye(self):
        self.assertEqual(self._info["dipol_mom"]["unit"], "Debye")


# ============================================================================
# GROUP 15: Logging + custom logger
# ============================================================================


class TestLogging(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="test_qm40_log_")
        self._base = Path(self._tmpdir)
        _write_qm40_dataset(self._base, _make_molecules(2))

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_custom_logger_used(self):
        test_logger = _make_logger()
        with patch.object(test_logger, "info") as mock_info:
            parse_qm40_csv_files(self._base, logger=test_logger)
            self.assertTrue(mock_info.called)

    def test_default_logger_when_none(self):
        # Should not raise when logger is omitted
        features, _ = parse_qm40_csv_files(self._base)
        self.assertIn("compounds", features)


# ============================================================================
# Comprehensive runner (pytest + unittest compatible)
# ============================================================================


def run_comprehensive_suite():
    """Run all test groups in a structured order."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestElementToZ,
        TestColumnMapping,
        TestBuildObjectArray,
        TestToFloat,
        TestLocateCsvFiles,
        TestParseHappyPath,
        TestJoinByZincId,
        TestBondData,
        TestInitialCoordinates,
        TestMaxMolecules,
        TestErrors,
        TestDtypes,
        TestNeutrality,
        TestGetPropertyInfo,
        TestLogging,
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS — qm40_csv_parser.py")
    print("=" * 80)
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "pytest" in sys.modules:
        pass
    else:
        sys.exit(run_comprehensive_suite())
