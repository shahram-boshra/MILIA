"""Structure inputs: molecules with given coordinates, built as the model's training dataset built its own.

A trained model records how its dataset built molecules (``data_info.molecule_creation``:
creation strategy, identifier keys, coordinate units, the charges seen, the converter switches).
:class:`StructureBuilder` builds a structure with that record through
``MoleculeDataConverter.convert_inputs``, the creation and featurization steps of training, so a
structure taken from the training data gives the training graph (``x``, ``edge_index``,
``edge_attr``, ``pos``, ``z``).

The molecular charge (RDKit ``DetermineBonds`` "requires the overall charge on the molecule"):

* identifier-based creation reads it from the identifier, as training did; a stated charge must
  agree with it;
* coordinate-based creation determines bonds for it, so the structure must state it.

Either way the charge must be one the training molecules were built with
(``molecule_creation.charges``; OECD (Q)SAR principle 3, a defined domain of applicability). A
refused structure is reported with its reason (:class:`~milia_pipeline.exceptions.StructureInputError`).

Readers: :func:`read_structures` reads XYZ (multi-frame; extended-XYZ ``key=value`` comment
pairs ``charge``, ``inchi``, ``smiles``, ``name``) and SDF/MOL with every hydrogen kept
(``removeHs=False``; RDKit removes them by default) and its 3D conformer (a 2D record is refused).
Coordinates are in ångström.
"""

from __future__ import annotations

import logging
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from milia_pipeline.exceptions import StructureInputError

__all__ = [
    "IDENTIFIER_TYPES",
    "STRUCTURE_SUFFIXES",
    "StructureBuilder",
    "StructureInput",
    "read_sdf",
    "read_structures",
    "read_xyz",
]

IDENTIFIER_TYPES = ("inchi", "smiles")
STRUCTURE_SUFFIXES = (".xyz", ".sdf", ".mol")
_XYZ_KEYS = ("charge", "inchi", "smiles", "name")


@dataclass(frozen=True)
class StructureInput:
    """One molecule with given coordinates (ångström), in the order of its atomic numbers.

    ``charge``: the molecular charge (required by coordinate-based creation); ``identifier`` and
    ``identifier_type`` (``inchi`` / ``smiles``): its identifier (required by identifier-based
    creation); ``name``: a label for reports.
    """

    atomic_numbers: tuple[int, ...]
    coordinates: tuple[tuple[float, float, float], ...]
    charge: int | None = None
    identifier: str | None = None
    identifier_type: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.atomic_numbers:
            raise ValueError("A structure needs at least one atom")
        if len(self.coordinates) != len(self.atomic_numbers) or any(
            len(xyz) != 3 for xyz in self.coordinates
        ):
            raise ValueError(
                f"{len(self.atomic_numbers)} atoms need {len(self.atomic_numbers)} coordinate "
                f"triples, got {[len(xyz) for xyz in self.coordinates]}"
            )
        if not all(1 <= int(z) <= 118 for z in self.atomic_numbers):
            raise ValueError(f"Atomic numbers must be 1-118, got {list(self.atomic_numbers)}")
        if not np.all(np.isfinite(np.asarray(self.coordinates, dtype=float))):
            raise ValueError("Coordinates must be finite")
        if (self.identifier is None) != (self.identifier_type is None):
            raise ValueError("identifier and identifier_type are given together")
        if self.identifier_type is not None and self.identifier_type not in IDENTIFIER_TYPES:
            raise ValueError(
                f"identifier_type must be one of {IDENTIFIER_TYPES}, got {self.identifier_type!r}"
            )
        if self.charge is not None and not isinstance(self.charge, int):
            raise ValueError(f"charge must be an integer, got {self.charge!r}")

    @classmethod
    def create(
        cls,
        atomic_numbers: Sequence[int],
        coordinates: Any,
        charge: int | None = None,
        identifier: str | None = None,
        identifier_type: str | None = None,
        name: str | None = None,
    ) -> StructureInput:
        """A structure from any sequence of atomic numbers and an ``(N, 3)`` array-like."""
        array = np.asarray(coordinates, dtype=float)
        if array.ndim != 2 or array.shape[1:] != (3,):
            raise ValueError(f"coordinates must have shape (N, 3), got {array.shape}")
        return cls(
            atomic_numbers=tuple(int(z) for z in atomic_numbers),
            coordinates=tuple(tuple(float(v) for v in row) for row in array),
            charge=None if charge is None else int(charge),
            identifier=identifier,
            identifier_type=identifier_type,
            name=name,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON form (a serving request instance; a package's structure example)."""
        return {
            "atomic_numbers": list(self.atomic_numbers),
            "coordinates": [list(xyz) for xyz in self.coordinates],
            "charge": self.charge,
            "identifier": self.identifier,
            "identifier_type": self.identifier_type,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StructureInput:
        """The structure of :meth:`to_dict`'s form (``coordinates`` in ångström).

        Raises:
            ValueError: An unknown key, or a field :class:`StructureInput` refuses.
        """
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown structure fields {sorted(unknown)}")
        if "atomic_numbers" not in data or "coordinates" not in data:
            raise ValueError("a structure needs atomic_numbers and coordinates")
        return cls.create(**data)

    @classmethod
    def from_mol(cls, mol: Any, name: str | None = None) -> StructureInput:
        """A structure from an RDKit molecule with a 3D conformer and all its hydrogens.

        Charge: the sum of its formal charges; identifier: its standard InChI.
        """
        from rdkit import Chem

        if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
            raise ValueError("the molecule has no 3D coordinates")
        inchi = Chem.MolToInchi(mol) or None
        return cls.create(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()],
            mol.GetConformer().GetPositions(),
            charge=Chem.GetFormalCharge(mol),
            identifier=inchi,
            identifier_type="inchi" if inchi else None,
            name=name,
        )


def read_xyz(path: str | Path) -> list[StructureInput]:
    """The structures of an XYZ file (one per frame).

    Each frame: an atom count, a comment line, then ``symbol x y z`` per atom (ångström). The
    comment line may carry extended-XYZ ``key=value`` pairs; ``charge``, ``inchi``, ``smiles`` and
    ``name`` are read, other keys ignored.

    Raises:
        ValueError: A malformed frame (named by its position), never skipped.
    """
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    lines = Path(path).read_text().splitlines()
    structures: list[StructureInput] = []
    position = 0
    while position < len(lines):
        if not lines[position].strip():
            position += 1
            continue
        frame = len(structures)
        try:
            count = int(lines[position].split()[0])
            comment = lines[position + 1] if position + 1 < len(lines) else ""
            atom_lines = lines[position + 2 : position + 2 + count]
            if count < 1 or len(atom_lines) != count:
                raise ValueError(f"{count} atoms declared, {len(atom_lines)} atom lines")
            numbers, coordinates = [], []
            for line in atom_lines:
                fields = line.split()
                symbol = fields[0]
                numbers.append(int(symbol) if symbol.isdigit() else table.GetAtomicNumber(symbol))
                coordinates.append([float(value) for value in fields[1:4]])
            pairs = _comment_pairs(comment)
            identifier_type = next((key for key in IDENTIFIER_TYPES if key in pairs), None)
            structures.append(
                StructureInput.create(
                    numbers,
                    coordinates,
                    charge=int(pairs["charge"]) if "charge" in pairs else None,
                    identifier=pairs.get(identifier_type) if identifier_type else None,
                    identifier_type=identifier_type,
                    name=pairs.get("name") or f"{Path(path).name}:{frame}",
                )
            )
        except (IndexError, ValueError, RuntimeError) as error:
            raise ValueError(f"{path}: frame {frame} is not a valid XYZ frame: {error}") from error
        position += 2 + count
    if not structures:
        raise ValueError(f"{path}: no XYZ frame")
    return structures


def _comment_pairs(comment: str) -> dict[str, str]:
    pairs = {}
    for token in shlex.split(comment):
        key, sep, value = token.partition("=")
        if sep and key.lower() in _XYZ_KEYS:
            pairs[key.lower()] = value
    return pairs


def read_sdf(path: str | Path) -> list[StructureInput]:
    """The structures of an SDF or MOL file, every hydrogen kept (``removeHs=False``).

    Raises:
        ValueError: A record RDKit cannot read, or one without 3D coordinates (named by its
            position), never skipped.
    """
    from rdkit import Chem

    path = Path(path)
    if path.suffix.lower() == ".mol":
        records = [Chem.MolFromMolFile(str(path), removeHs=False)]
    else:
        records = list(Chem.SDMolSupplier(str(path), removeHs=False))
    structures = []
    for index, mol in enumerate(records):
        if mol is None:
            raise ValueError(f"{path}: record {index} cannot be read by RDKit")
        name = mol.GetProp("_Name") if mol.HasProp("_Name") and mol.GetProp("_Name") else None
        try:
            structures.append(StructureInput.from_mol(mol, name=name or f"{path.name}:{index}"))
        except ValueError as error:
            raise ValueError(f"{path}: record {index}: {error}") from error
    if not structures:
        raise ValueError(f"{path}: no record")
    return structures


def read_structures(path: str | Path) -> list[StructureInput]:
    """The structures of an XYZ, SDF or MOL file (by suffix)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".xyz":
        return read_xyz(path)
    if suffix in (".sdf", ".mol"):
        return read_sdf(path)
    raise ValueError(f"{path}: not a structure file ({', '.join(STRUCTURE_SUFFIXES)})")


class StructureBuilder:
    """Builds structure inputs with a trained model's record of its dataset's molecule creation.

    Args:
        molecule_creation: ``data_info.molecule_creation`` of the checkpoint (with ``charges``
            and ``featurization``; the input contract refuses structures without them).
        structural_features_config: ``data_info.structural_features_config``.
    """

    def __init__(
        self,
        molecule_creation: Mapping[str, Any],
        structural_features_config: Mapping[str, Any] | None,
        logger: logging.Logger | None = None,
    ) -> None:
        from milia_pipeline.config.config_containers import (
            DatasetConfig,
            FilterConfig,
            ProcessingConfig,
        )
        from milia_pipeline.datasets import registry
        from milia_pipeline.molecules.molecule_converter_core import MoleculeDataConverter

        self.logger = logger or logging.getLogger(__name__)
        self.dataset_type = molecule_creation["dataset_type"]
        self.strategy = molecule_creation["strategy"]
        self.identifier_keys = [tuple(pair) for pair in molecule_creation["identifier_keys"]]
        self.coordinate_units = molecule_creation["coordinate_units"]
        self.charges = list(molecule_creation["charges"])
        self.switches = dict(molecule_creation["featurization"])

        dataset_config = DatasetConfig(dataset_type=self.dataset_type)
        filter_config = FilterConfig()
        processing_config = ProcessingConfig(scalar_graph_targets=[])  # a structure has no labels
        handler = registry.get(self.dataset_type).create_handler(
            dataset_config, filter_config, processing_config, self.logger
        )
        self._converter = MoleculeDataConverter(
            dataset_handler=handler,
            dataset_config=dataset_config,
            filter_config=filter_config,
            processing_config=processing_config,
            logger=self.logger,
            structural_features_config=dict(structural_features_config or {}),
        )
        if self.coordinate_units == "bohr":
            from milia_pipeline.config.config_constants import get_bohr_to_angstrom

            self._from_angstrom = 1.0 / get_bohr_to_angstrom()
        else:
            self._from_angstrom = 1.0

    def build(self, structures: Sequence[StructureInput]) -> list[Any]:
        """The graphs of ``structures``, in order.

        Raises:
            StructureInputError: Every refused structure with its reason (none is built then).
        """
        graphs, refusals = [], []
        for index, structure in enumerate(structures):
            try:
                graphs.append(self.build_one(structure, index))
            except StructureInputError as error:
                refusals.extend(error.refusals)
        if refusals:
            raise StructureInputError(refusals)
        return graphs

    def build_one(self, structure: StructureInput, index: int = 0) -> Any:
        """The graph of one structure.

        Raises:
            StructureInputError: The structure is refused (its reason).
        """
        from milia_pipeline.exceptions import HandlerError, MoleculeProcessingError

        identifier, identifier_type = None, None
        if self.strategy == "identifier_coordinate_based":
            identifier, identifier_type = self._identifier(structure, index)
        elif structure.charge is None:
            self._refuse(
                index,
                f"{self.dataset_type} builds molecules from coordinates, which needs the "
                f"molecular charge; state it (training charges: {self.charges})",
            )
        try:
            graph = self._converter.convert_inputs(
                molecule_index=index,
                atomic_numbers=np.asarray(structure.atomic_numbers, dtype=np.int64),
                coordinates=np.asarray(structure.coordinates, dtype=np.float64)
                * self._from_angstrom,
                molecular_charge=(
                    structure.charge if self.strategy == "coordinate_based" else None
                ),
                switches=self.switches,
                identifier=identifier,
                identifier_type=identifier_type,
            )
        except (MoleculeProcessingError, HandlerError) as error:
            self._refuse(index, f"the molecule cannot be built: {error}")
        built = int(graph.molecular_charge.reshape(-1)[0])
        if structure.charge is not None and structure.charge != built:
            self._refuse(
                index,
                f"stated charge {structure.charge} differs from the identifier's charge {built}",
            )
        if built not in self.charges:
            self._refuse(
                index,
                f"charge {built} is outside the charges the model was trained on {self.charges}",
            )
        return graph

    def _identifier(self, structure: StructureInput, index: int) -> tuple[str, str]:
        """The structure's identifier as the dataset's first identifier kind (training reads it)."""
        if structure.identifier is None:
            self._refuse(
                index,
                f"{self.dataset_type} reads bonds from an identifier "
                f"({', '.join(kind for _, kind in self.identifier_keys)}); give one with the "
                "coordinates (an SDF/MOL record carries its InChI)",
            )
        kinds = [kind for _, kind in self.identifier_keys]
        primary = kinds[0]
        if structure.identifier_type == primary:
            return structure.identifier, primary
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        try:
            mol = (
                Chem.MolFromSmiles(structure.identifier)
                if structure.identifier_type == "smiles"
                else Chem.MolFromInchi(structure.identifier)
            )
            converted = None
            if mol is not None:
                converted = Chem.MolToInchi(mol) if primary == "inchi" else Chem.MolToSmiles(mol)
        finally:
            RDLogger.EnableLog("rdApp.*")
        if not converted:
            self._refuse(
                index,
                f"its {structure.identifier_type} {structure.identifier!r} cannot be converted to "
                f"{primary}, the identifier {self.dataset_type} builds from",
            )
        return converted, primary

    @staticmethod
    def _refuse(index: int, reason: str) -> None:
        raise StructureInputError([(index, reason)])
