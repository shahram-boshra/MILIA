"""
QM40 CSV Format Parser
──────────────────────

Parser for the QM40 dataset's CSV file format.

QM40 (Madushanka, Moura Jr. & Kraka, *Scientific Data* 11:1376, 2024;
DOI 10.1038/s41597-024-04206-y) ships as three CSV files inside
``QM40_dataset.zip`` (Figshare DOI 10.6084/m9.figshare.25993060). All
162,954 molecules are drug-like ZINC structures optimized at the
B3LYP/6-31G(2df,p) level of theory in Gaussian16. The dataset is
**neutral only** — anions and cations are explicitly excluded — so every
molecule is a charge-neutral singlet ground state.

QM40 CSV layout (verified by line-by-line inspection of the real archive)
─────────────────────────────────────────────────────────────────────────
The in-zip folder name contains a SPACE: ``QM40 dataset/``. macOS packed the
archive, so a ``__MACOSX/`` tree of AppleDouble (``._*``) resource forks is
also present and MUST be ignored.

1. ``QM40_main.csv`` — one row per molecule (18 columns):
   ``Zinc_id, smile, Internal_E(0K), HOMO, LUMO, HL_gap, Polarizability,
   spatial extent, dipol_mom, ZPE, rot1, rot2, rot3, Inter_E(298), Enthalpy,
   Free_E, CV, Entropy``
   NOTE: the column ``spatial extent`` contains a SPACE (the paper prints it as
   ``spatial_extent``); it is sanitized to ``spatial_extent`` for the NPZ key.
   NOTE: ``HL_gap`` is stored as ``HOMO - LUMO`` (i.e. negative).

2. ``QM40_xyz.csv`` — one row per ATOM (10 columns), ``Zinc_id`` repeated:
   ``Zinc_id, smile, atom, init_x, init_y, init_z, final_x, final_y, final_z,
   charge`` where ``atom`` is an element SYMBOL and ``charge`` is the per-atom
   Mulliken charge. The optimized geometry (``final_*``) is the primary
   coordinate set; ``init_*`` is the pre-optimization geometry.

3. ``QM40_bond.csv`` — one row per BOND (7 columns), ``Zinc_id`` repeated:
   ``Zinc_id, smile, atom1, atom2, bond, tag, lmod`` where ``atom1``/``atom2``
   are **1-based** atom indices into that molecule's atom list and ``lmod`` is
   the local vibrational mode stretching force constant (mDyn/Å), QM40's
   distinguishing bond-strength feature.

CRITICAL: the three files are NOT row-aligned; they MUST be joined by
``Zinc_id``. The atom order for a molecule is the row order of its atoms in
``QM40_xyz.csv``; bond atom indices refer to that same order.

Property units (paper Table 2)
─────────────────────────────────────────────────────────────────────────
Internal_E_0K, HOMO, LUMO, HL_gap, Inter_E_298, Enthalpy, Free_E : Hartree
Polarizability : Bohr³   spatial_extent : Bohr²   dipol_mom : Debye
ZPE : kcal/mol   rot1/rot2/rot3 : GHz   CV, Entropy : cal/(mol·K)

Author: milia Pipeline Team
Version: 1.0.0
"""

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from milia_pipeline.exceptions import DataProcessingError

logger = logging.getLogger(__name__)


# Atomic-number lookup, scoped to QM40's documented element set
# (C, O, N, S, F, Cl + H — paper §Background and Table 5). Restricting the map
# to these elements makes any out-of-set symbol fail loudly rather than be
# silently mis-mapped, which is the correct behaviour for a dataset that is
# *defined* over exactly these elements.
ELEMENT_TO_Z: dict[str, int] = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "S": 16,
    "Cl": 17,
}

# Mapping from the EXACT QM40_main.csv column headers to sanitized NPZ keys.
# The left-hand side must match the real header byte-for-byte (note the
# parentheses and the SPACE in "spatial extent").
QM40_MAIN_COLUMN_TO_KEY: dict[str, str] = {
    "Internal_E(0K)": "Internal_E_0K",
    "HOMO": "HOMO",
    "LUMO": "LUMO",
    "HL_gap": "HL_gap",
    "Polarizability": "Polarizability",
    "spatial extent": "spatial_extent",
    "dipol_mom": "dipol_mom",
    "ZPE": "ZPE",
    "rot1": "rot1",
    "rot2": "rot2",
    "rot3": "rot3",
    "Inter_E(298)": "Inter_E_298",
    "Enthalpy": "Enthalpy",
    "Free_E": "Free_E",
    "CV": "CV",
    "Entropy": "Entropy",
}

# Sanitized scalar property keys, in canonical order.
QM40_SCALAR_KEYS: list[str] = list(QM40_MAIN_COLUMN_TO_KEY.values())

# Identifier / structural column names (xyz + bond files).
_ID_COLUMN = "Zinc_id"
_SMILES_COLUMN = "smile"


def _locate_csv_files(csv_dir: Path) -> tuple[Path, Path, Path]:
    """
    Locate the QM40 main / xyz / bond CSV files under an extraction directory.

    AppleDouble resource forks (``._*``) and ``__MACOSX`` directories created by
    macOS zip are excluded. Files are matched by case-insensitive substring
    ("main", "xyz", "bond") so the function is robust to minor naming variants.

    Args:
        csv_dir: Directory containing the extracted QM40 CSV files.

    Returns:
        Tuple of (main_csv, xyz_csv, bond_csv) paths.

    Raises:
        DataProcessingError: If any of the three CSV files cannot be located.
    """
    candidates = [
        p
        for p in csv_dir.rglob("*.csv")
        if "__MACOSX" not in p.parts and not p.name.startswith("._")
    ]

    def _find(token: str) -> Path | None:
        matches = sorted(p for p in candidates if token in p.name.lower())
        return matches[0] if matches else None

    main_csv = _find("main")
    xyz_csv = _find("xyz")
    bond_csv = _find("bond")

    missing = [
        name
        for name, path in (("main", main_csv), ("xyz", xyz_csv), ("bond", bond_csv))
        if path is None
    ]
    if missing:
        raise DataProcessingError(
            f"QM40 CSV file(s) not found under {csv_dir}: missing {missing}",
            file_path=str(csv_dir),
            operation="qm40_csv_parsing",
            details=f"Discovered CSVs: {[p.name for p in candidates]}",
        )

    return main_csv, xyz_csv, bond_csv  # type: ignore[return-value]


def _to_float(value: str) -> float:
    """Parse a CSV cell to float; empty/whitespace cells become NaN."""
    text = (value or "").strip()
    if not text:
        return float("nan")
    return float(text)


def _read_main_csv(
    main_csv: Path, max_molecules: int | None, logger: logging.Logger
) -> tuple[list[str], dict[str, str], dict[str, dict[str, float]], int]:
    """
    Read QM40_main.csv.

    Returns:
        Tuple of:
        - ordered_ids: molecule ids in file order (truncated to max_molecules)
        - smiles_by_id: {Zinc_id: SMILES}
        - scalars_by_id: {Zinc_id: {sanitized_key: float}}
        - total_rows: total molecule rows present in the file (pre-truncation)
    """
    ordered_ids: list[str] = []
    smiles_by_id: dict[str, str] = {}
    scalars_by_id: dict[str, dict[str, float]] = {}
    total_rows = 0

    with open(main_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        header = reader.fieldnames or []
        missing_cols = [c for c in QM40_MAIN_COLUMN_TO_KEY if c not in header]
        if _ID_COLUMN not in header or missing_cols:
            raise DataProcessingError(
                f"QM40_main.csv missing expected columns: "
                f"id_present={_ID_COLUMN in header}, missing_scalars={missing_cols}",
                file_path=str(main_csv),
                operation="qm40_csv_parsing",
                details=f"Header seen: {header}",
            )

        for row in reader:
            total_rows += 1
            if max_molecules is not None and len(ordered_ids) >= max_molecules:
                # Keep counting total_rows would require reading the whole file;
                # stop early since selection is the first N in file order.
                total_rows = len(ordered_ids)
                break

            zinc_id = (row.get(_ID_COLUMN) or "").strip()
            if not zinc_id:
                continue

            ordered_ids.append(zinc_id)
            smiles_by_id[zinc_id] = (row.get(_SMILES_COLUMN) or "").strip()
            scalars_by_id[zinc_id] = {
                npz_key: _to_float(row.get(csv_col, ""))
                for csv_col, npz_key in QM40_MAIN_COLUMN_TO_KEY.items()
            }

    if not ordered_ids:
        raise DataProcessingError(
            "QM40_main.csv contained no molecule rows",
            file_path=str(main_csv),
            operation="qm40_csv_parsing",
        )

    logger.info(f"main.csv: selected {len(ordered_ids)} molecule(s)")
    return ordered_ids, smiles_by_id, scalars_by_id, total_rows


def _read_xyz_csv(
    xyz_csv: Path,
    selected_ids: set[str],
    include_initial_coordinates: bool,
    logger: logging.Logger,
) -> dict[str, dict[str, list]]:
    """
    Stream QM40_xyz.csv, accumulating per-molecule atom data for selected ids.

    Returns:
        {Zinc_id: {'atoms': [Z...], 'coordinates': [[x,y,z]...],
                   'initial_coordinates': [[x,y,z]...] (optional),
                   'Qmulliken': [q...]}}
        Atom order follows the row order in the file.

    Raises:
        DataProcessingError: On unknown element symbols (per offending molecule;
            re-raised so the caller can attribute the failure to a molecule).
    """
    per_mol: dict[str, dict[str, list]] = {}

    with open(xyz_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        required = {_ID_COLUMN, "atom", "final_x", "final_y", "final_z", "charge"}
        if not required.issubset(header):
            raise DataProcessingError(
                f"QM40_xyz.csv missing expected columns; need {sorted(required)}",
                file_path=str(xyz_csv),
                operation="qm40_csv_parsing",
                details=f"Header seen: {header}",
            )

        for row in reader:
            zinc_id = (row.get(_ID_COLUMN) or "").strip()
            if zinc_id not in selected_ids:
                continue

            bucket = per_mol.setdefault(
                zinc_id,
                {"atoms": [], "coordinates": [], "initial_coordinates": [], "Qmulliken": []},
            )

            symbol = (row.get("atom") or "").strip()
            z = ELEMENT_TO_Z.get(symbol)
            if z is None:
                raise DataProcessingError(
                    f"Unknown element '{symbol}' for molecule {zinc_id}",
                    file_path=str(xyz_csv),
                    operation="qm40_csv_parsing",
                    details=f"QM40 supports {sorted(ELEMENT_TO_Z)}",
                )

            bucket["atoms"].append(z)
            bucket["coordinates"].append(
                [_to_float(row["final_x"]), _to_float(row["final_y"]), _to_float(row["final_z"])]
            )
            if include_initial_coordinates:
                bucket["initial_coordinates"].append(
                    [_to_float(row["init_x"]), _to_float(row["init_y"]), _to_float(row["init_z"])]
                )
            bucket["Qmulliken"].append(_to_float(row["charge"]))

    logger.info(f"xyz.csv: assembled atom data for {len(per_mol)} molecule(s)")
    return per_mol


def _read_bond_csv(
    bond_csv: Path, selected_ids: set[str], logger: logging.Logger
) -> dict[str, dict[str, list]]:
    """
    Stream QM40_bond.csv, accumulating per-molecule bond data for selected ids.

    ``atom1``/``atom2`` are converted from 1-based (file convention) to 0-based
    atom indices. The redundant ``bond`` label column (derivable from atoms +
    indices) is not stored.

    Returns:
        {Zinc_id: {'bond_atom1_idx': [int...], 'bond_atom2_idx': [int...],
                   'bond_tag': [str...], 'bond_lmod_ka': [float...]}}
    """
    per_mol: dict[str, dict[str, list]] = {}

    with open(bond_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        required = {_ID_COLUMN, "atom1", "atom2", "tag", "lmod"}
        if not required.issubset(header):
            raise DataProcessingError(
                f"QM40_bond.csv missing expected columns; need {sorted(required)}",
                file_path=str(bond_csv),
                operation="qm40_csv_parsing",
                details=f"Header seen: {header}",
            )

        for row in reader:
            zinc_id = (row.get(_ID_COLUMN) or "").strip()
            if zinc_id not in selected_ids:
                continue

            bucket = per_mol.setdefault(
                zinc_id,
                {"bond_atom1_idx": [], "bond_atom2_idx": [], "bond_tag": [], "bond_lmod_ka": []},
            )
            # 1-based -> 0-based
            bucket["bond_atom1_idx"].append(int(row["atom1"]) - 1)
            bucket["bond_atom2_idx"].append(int(row["atom2"]) - 1)
            bucket["bond_tag"].append((row.get("tag") or "").strip())
            bucket["bond_lmod_ka"].append(_to_float(row["lmod"]))

    logger.info(f"bond.csv: assembled bond data for {len(per_mol)} molecule(s)")
    return per_mol


def parse_qm40_csv_files(
    csv_dir: Path,
    max_molecules: int | None = None,
    include_bond_data: bool = True,
    include_initial_coordinates: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Parse the three QM40 CSV files in a directory and build feature arrays.

    This is the main entry point for QM40 preprocessing. The three CSVs are
    joined by ``Zinc_id`` (they are NOT row-aligned). The molecule set and its
    order are defined by ``QM40_main.csv``; ``max_molecules`` selects the first
    N molecules in that order (deterministic).

    Args:
        csv_dir: Directory containing the extracted QM40 CSV files.
        max_molecules: Maximum number of molecules to parse (None = all).
        include_bond_data: Store per-bond local-mode force constants and indices.
        include_initial_coordinates: Store pre-optimization coordinates.
        logger: Logger instance (uses module logger if None).

    Returns:
        Tuple of (features_dict, metadata_dict):
        - features_dict: numpy arrays for NPZ creation. Ragged per-molecule
          arrays use ``dtype=object``; scalar properties use ``np.float64``.
          Keys: ``compounds`` (Zinc_id), ``smiles``, ``atoms``, ``coordinates``
          (optimized), ``Qmulliken``, the 16 sanitized scalar keys, and
          optionally ``initial_coordinates`` and the ``bond_*`` arrays.
        - metadata_dict: parsing statistics and provenance.

    Raises:
        DataProcessingError: If the CSV files cannot be located/parsed or if no
            molecule is successfully assembled.
    """
    if logger is None:
        logger = globals()["logger"]

    csv_dir = Path(csv_dir)
    main_csv, xyz_csv, bond_csv = _locate_csv_files(csv_dir)
    logger.info("Parsing commenced...")

    ordered_ids, smiles_by_id, scalars_by_id, total_rows = _read_main_csv(
        main_csv, max_molecules, logger
    )
    selected = set(ordered_ids)

    xyz_by_id = _read_xyz_csv(xyz_csv, selected, include_initial_coordinates, logger)
    bond_by_id = _read_bond_csv(bond_csv, selected, logger) if include_bond_data else {}

    # Collection lists, built in main.csv order
    compounds: list[str] = []
    smiles_list: list[str] = []
    atoms_list: list[np.ndarray] = []
    coordinates_list: list[np.ndarray] = []
    initial_coordinates_list: list[np.ndarray] = []
    qmulliken_list: list[np.ndarray] = []
    bond_a1_list: list[np.ndarray] = []
    bond_a2_list: list[np.ndarray] = []
    bond_tag_list: list[np.ndarray] = []
    bond_lmod_list: list[np.ndarray] = []
    scalar_props: dict[str, list[float]] = {key: [] for key in QM40_SCALAR_KEYS}

    parsed_count = 0
    failed_count = 0
    failed_ids: list[str] = []
    dropped_bonds = 0

    for zinc_id in ordered_ids:
        try:
            atom_data = xyz_by_id.get(zinc_id)
            if not atom_data or not atom_data["atoms"]:
                raise DataProcessingError(
                    f"No atom rows in xyz.csv for molecule {zinc_id}",
                    operation="qm40_csv_parsing",
                )

            atoms = np.array(atom_data["atoms"], dtype=np.int32)
            coordinates = np.array(atom_data["coordinates"], dtype=np.float64)
            qmulliken = np.array(atom_data["Qmulliken"], dtype=np.float64)
            num_atoms = atoms.shape[0]

            if coordinates.shape != (num_atoms, 3) or qmulliken.shape[0] != num_atoms:
                raise DataProcessingError(
                    f"Inconsistent atom/coordinate/charge counts for {zinc_id}",
                    operation="qm40_csv_parsing",
                )

            # Per-molecule assembly succeeded — commit core arrays
            compounds.append(zinc_id)
            smiles_list.append(smiles_by_id.get(zinc_id, ""))
            atoms_list.append(atoms)
            coordinates_list.append(coordinates)
            qmulliken_list.append(qmulliken)

            if include_initial_coordinates:
                init = np.array(atom_data["initial_coordinates"], dtype=np.float64)
                # Fall back to an empty (0, 3) array if initial coords are absent
                if init.size == 0:
                    init = np.empty((0, 3), dtype=np.float64)
                initial_coordinates_list.append(init)

            if include_bond_data:
                bdata = bond_by_id.get(zinc_id)
                if bdata:
                    a1 = np.array(bdata["bond_atom1_idx"], dtype=np.int64)
                    a2 = np.array(bdata["bond_atom2_idx"], dtype=np.int64)
                    tags = np.array(bdata["bond_tag"], dtype=object)
                    ka = np.array(bdata["bond_lmod_ka"], dtype=np.float64)
                    # Drop any bond whose index falls outside this molecule's atoms
                    valid = (a1 >= 0) & (a1 < num_atoms) & (a2 >= 0) & (a2 < num_atoms)
                    if not bool(np.all(valid)):
                        dropped_bonds += int((~valid).sum())
                        a1, a2, tags, ka = a1[valid], a2[valid], tags[valid], ka[valid]
                    bond_a1_list.append(a1)
                    bond_a2_list.append(a2)
                    bond_tag_list.append(tags)
                    bond_lmod_list.append(ka)
                else:
                    bond_a1_list.append(np.array([], dtype=np.int64))
                    bond_a2_list.append(np.array([], dtype=np.int64))
                    bond_tag_list.append(np.array([], dtype=object))
                    bond_lmod_list.append(np.array([], dtype=np.float64))

            sc = scalars_by_id.get(zinc_id, {})
            for key in QM40_SCALAR_KEYS:
                scalar_props[key].append(sc.get(key, float("nan")))

            parsed_count += 1

        except Exception as e:  # noqa: BLE001 — per-molecule isolation, like QM9 parser
            failed_count += 1
            failed_ids.append(zinc_id)
            logger.warning(f"Failed to assemble molecule {zinc_id}: {e}")
            if failed_count > 100 and failed_count > parsed_count * 0.1:
                raise DataProcessingError(
                    f"Too many assembly failures ({failed_count}/{parsed_count + failed_count})",
                    operation="qm40_csv_parsing",
                    details=f"First failures: {failed_ids[:10]}",
                ) from e

    if parsed_count == 0:
        raise DataProcessingError(
            "No QM40 molecules successfully assembled",
            file_path=str(csv_dir),
            operation="qm40_csv_parsing",
        )

    logger.info(
        f"✓ Parsing complete: {parsed_count} molecule(s) assembled ({failed_count} failure(s))"
    )

    features: dict[str, np.ndarray] = {
        # Required core features
        "compounds": np.array(compounds, dtype=object),
        "atoms": np.array(atoms_list, dtype=object),
        "coordinates": np.array(coordinates_list, dtype=object),
        # Per-atom node feature (canonical Mulliken key, matches QM9/DFT)
        "Qmulliken": np.array(qmulliken_list, dtype=object),
        # Tracking label (not used for graph construction — coordinate_based)
        "smiles": np.array(smiles_list, dtype=object),
    }

    if include_initial_coordinates:
        features["initial_coordinates"] = np.array(initial_coordinates_list, dtype=object)

    if include_bond_data:
        features["bond_atom1_idx"] = np.array(bond_a1_list, dtype=object)
        features["bond_atom2_idx"] = np.array(bond_a2_list, dtype=object)
        features["bond_tag"] = np.array(bond_tag_list, dtype=object)
        features["bond_lmod_ka"] = np.array(bond_lmod_list, dtype=object)

    for key in QM40_SCALAR_KEYS:
        features[key] = np.array(scalar_props[key], dtype=np.float64)

    metadata: dict[str, Any] = {
        "num_molecules_parsed": parsed_count,
        "num_molecules_failed": failed_count,
        "total_molecules_in_main": total_rows,
        "source_format": "qm40_csv",
        "property_names": list(QM40_SCALAR_KEYS),
        "has_mulliken_charges": True,
        "has_bond_data": bool(include_bond_data),
        "has_initial_coordinates": bool(include_initial_coordinates),
        "coordinate_units": "angstrom",
        "energy_units": "hartree",
        "level_of_theory": "B3LYP/6-31G(2df,p)",
        "elements_supported": sorted(ELEMENT_TO_Z),
    }
    if include_bond_data and dropped_bonds:
        metadata["dropped_out_of_range_bonds"] = dropped_bonds
    if failed_ids:
        metadata["failed_ids_sample"] = failed_ids[:10]

    return features, metadata


def get_qm40_property_info() -> dict[str, dict[str, str]]:
    """
    Get information about QM40 scalar properties (units and descriptions).

    Returns:
        Dictionary mapping sanitized property keys to ``{unit, description}``.
    """
    return {
        "Internal_E_0K": {"unit": "Hartree", "description": "Internal energy at 0 K"},
        "HOMO": {"unit": "Hartree", "description": "HOMO energy"},
        "LUMO": {"unit": "Hartree", "description": "LUMO energy"},
        "HL_gap": {"unit": "Hartree", "description": "HOMO-LUMO gap (HOMO - LUMO)"},
        "Polarizability": {"unit": "Bohr³", "description": "Isotropic polarizability"},
        "spatial_extent": {"unit": "Bohr²", "description": "Electronic spatial extent"},
        "dipol_mom": {"unit": "Debye", "description": "Dipole moment"},
        "ZPE": {"unit": "kcal/mol", "description": "Zero point energy"},
        "rot1": {"unit": "GHz", "description": "Rotational constant 1"},
        "rot2": {"unit": "GHz", "description": "Rotational constant 2"},
        "rot3": {"unit": "GHz", "description": "Rotational constant 3"},
        "Inter_E_298": {"unit": "Hartree", "description": "Internal energy at 298.15 K"},
        "Enthalpy": {"unit": "Hartree", "description": "Enthalpy at 298.15 K"},
        "Free_E": {"unit": "Hartree", "description": "Free energy at 298.15 K"},
        "CV": {"unit": "cal/(mol·K)", "description": "Heat capacity at 298.15 K"},
        "Entropy": {"unit": "cal/(mol·K)", "description": "Entropy at 298.15 K"},
    }
