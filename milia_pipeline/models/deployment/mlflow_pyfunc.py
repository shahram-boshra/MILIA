"""A deployment package as an MLflow pyfunc model (the ``mlflow`` extra).

:class:`MiliaPackageModel` wraps a published package (:class:`~.serving.PackagedPredictor`): MLflow
logs it with Models from Code (:mod:`.mlflow_model`, a script that calls ``set_model``, so no
model object is pickled) and the package folder as the artifact ``package``. Loading verifies the
package (format, SHA-256, featurization) as the server does.

**Input** (one row per instance; the columns are those of the input kinds the package serves):

* ``smiles`` (string): an identifier instance, featurized as the training molecules were;
* ``atomic_numbers`` (array of long), ``coordinates`` (array of ``[x, y, z]`` in ångström),
  ``charge`` (long), ``identifier`` and ``identifier_type`` (string), ``name`` (string): a
  structure instance, built with its given coordinates (``StructureInstance`` of :mod:`.serving`).

A row with a ``smiles`` value is a SMILES; any other row is a structure. MLflow checks every
request against this schema (the model's signature) before :meth:`MiliaPackageModel.predict`.

**Output** (one row per instance): one column per output (the target names, else
``output_<i>``). A graph-level model gives one value per column; a node-level model gives each
instance's per-atom values (an array, in atom order) and an ``atomic_numbers`` column. A package
with conformal calibration adds ``<output>_lower`` and ``<output>_upper``; a bound that is not
finite (a calibration set too small for ``1 - alpha``) is null.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

# At run time, not only for type checking: MLflow reads predict's type hints (pandas is a MILIA
# dependency; this module is imported only with the mlflow extra)
import pandas as pd

from milia_pipeline.exceptions import DeploymentError
from milia_pipeline.models.capabilities import output_level
from milia_pipeline.models.input_contract import IDENTIFIER, STRUCTURE

__all__ = [
    "PACKAGE_ARTIFACT",
    "STRUCTURE_COLUMNS",
    "MiliaPackageModel",
    "input_example",
    "instances_from_frame",
    "output_columns",
    "predictions_frame",
    "pyfunc_signature",
    "served_kinds",
]

PACKAGE_ARTIFACT = "package"
SMILES_COLUMN = "smiles"
ATOMIC_NUMBERS_COLUMN = "atomic_numbers"
# StructureInstance's fields, in its order (serving.py)
STRUCTURE_COLUMNS = (
    "atomic_numbers",
    "coordinates",
    "charge",
    "identifier",
    "identifier_type",
    "name",
)


def served_kinds(manifest: dict[str, Any]) -> list[str]:
    """The instance kinds a package serves, as :class:`~.serving.PackagedPredictor` decides:
    identifiers when the model accepts them; structures when it accepts them and the package
    records a verified structure example."""
    accepted = (((manifest.get("signature") or {}).get("inputs") or {}).get("kinds") or {}).get(
        "accepted"
    ) or []
    kinds = []
    if IDENTIFIER in accepted:
        kinds.append(IDENTIFIER)
    if STRUCTURE in accepted and manifest.get("structure_example"):
        kinds.append(STRUCTURE)
    return kinds


def output_columns(manifest: dict[str, Any]) -> list[str]:
    """The output column names: the target names when there is one per output, unique, and not
    an input-side or derived column name; else ``output_<i>``."""
    outputs = (manifest.get("signature") or {}).get("outputs") or {}
    count = outputs.get("count")
    if not isinstance(count, int) or count < 1:
        raise DeploymentError("the package's signature records no output count")
    names = [str(name) for name in outputs.get("names") or []]
    derived = {f"{n}_{bound}" for n in names for bound in ("lower", "upper")}
    if (
        len(names) == count
        and len(set(names)) == count
        and not set(names) & ({ATOMIC_NUMBERS_COLUMN} | derived)
    ):
        return names
    return [f"output_{i}" for i in range(count)]


def _calibrated(manifest: dict[str, Any]) -> bool:
    return bool(manifest.get("calibration"))


def pyfunc_signature(manifest: dict[str, Any]) -> Any:
    """The MLflow ``ModelSignature`` of a package (see the module docstring).

    Raises:
        DeploymentError: The package serves no instance kind, or its output level is not served.
    """
    from mlflow.models import ModelSignature
    from mlflow.types import ColSpec, DataType, Schema
    from mlflow.types.schema import Array

    from milia_pipeline.models.deployment.serving import output_level_refusal

    refusal = output_level_refusal(manifest.get("task_type"))
    if refusal is not None:
        raise DeploymentError(refusal)
    kinds = served_kinds(manifest)
    if not kinds:
        raise DeploymentError(
            "the package serves neither SMILES nor structure instances (accepted input kinds: "
            f"{', '.join(_accepted(manifest)) or 'none'}; a structure needs a verified "
            "structure example)"
        )
    both = len(kinds) == 2
    inputs = []
    if IDENTIFIER in kinds:
        inputs.append(ColSpec(DataType.string, SMILES_COLUMN, required=not both))
    if STRUCTURE in kinds:
        inputs += [
            ColSpec(Array(DataType.long), "atomic_numbers", required=not both),
            ColSpec(Array(Array(DataType.double)), "coordinates", required=not both),
            ColSpec(DataType.long, "charge", required=False),
            ColSpec(DataType.string, "identifier", required=False),
            ColSpec(DataType.string, "identifier_type", required=False),
            ColSpec(DataType.string, "name", required=False),
        ]
    node = output_level(manifest.get("task_type")) == "node"
    value = Array(DataType.double) if node else DataType.double
    outputs = []
    for name in output_columns(manifest):
        outputs.append(ColSpec(value, name))
        if _calibrated(manifest):
            outputs += [
                ColSpec(value, f"{name}_lower", required=False),
                ColSpec(value, f"{name}_upper", required=False),
            ]
    if node:
        outputs.append(ColSpec(Array(DataType.long), ATOMIC_NUMBERS_COLUMN))
    return ModelSignature(inputs=Schema(inputs), outputs=Schema(outputs))


def _accepted(manifest: dict[str, Any]) -> list[str]:
    return (((manifest.get("signature") or {}).get("inputs") or {}).get("kinds") or {}).get(
        "accepted"
    ) or []


def input_example(manifest: dict[str, Any]) -> pd.DataFrame:
    """The package's verified examples as a request frame: its SMILES input example and its
    structure examples, for the kinds it serves."""
    kinds = served_kinds(manifest)
    rows: list[dict[str, Any]] = []
    if IDENTIFIER in kinds:
        rows += [{SMILES_COLUMN: s} for s in manifest.get("input_example") or ["C"]]
    if STRUCTURE in kinds:
        rows += [
            {column: example.get(column) for column in STRUCTURE_COLUMNS}
            for example in manifest.get("structure_example") or []
        ]
    columns = ([SMILES_COLUMN] if IDENTIFIER in kinds else []) + (
        list(STRUCTURE_COLUMNS) if STRUCTURE in kinds else []
    )
    return pd.DataFrame(rows, columns=columns)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(isinstance(value, float) and math.isnan(value))
    except TypeError:
        return False


def _plain(value: Any) -> Any:
    """A request value as Python lists and scalars (MLflow gives arrays as numpy arrays)."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def instances_from_frame(frame: pd.DataFrame) -> list[Any]:
    """Request instances from a pyfunc input frame: a SMILES string for a row with a ``smiles``
    value, else a structure mapping of the row's structure columns (missing ones left out).

    Raises:
        ValueError: A row holds neither a SMILES nor atomic numbers; the message names its index.
    """
    instances: list[Any] = []
    for index, row in enumerate(frame.to_dict(orient="records")):
        smiles = row.get(SMILES_COLUMN)
        if not _missing(smiles):
            instances.append(str(smiles))
            continue
        fields = {
            column: _plain(row[column])
            for column in STRUCTURE_COLUMNS
            if column in row and not _missing(row[column])
        }
        if "atomic_numbers" not in fields:
            raise ValueError(f"row {index}: neither smiles nor atomic_numbers is given")
        if "charge" in fields:
            fields["charge"] = int(fields["charge"])
        instances.append(fields)
    return instances


def _column(rows: Sequence[Any], j: int, node: bool) -> list[Any]:
    if node:
        return [[atom[j] for atom in instance] for instance in rows]
    return [row[j] for row in rows]


def predictions_frame(
    manifest: dict[str, Any],
    rows: Sequence[Any],
    intervals: dict[str, Any] | None,
    atomic_numbers: Sequence[Sequence[int]] | None,
) -> pd.DataFrame:
    """The output frame of :class:`~.serving.PackagedPredictor.predict_response`'s results."""
    node = output_level(manifest.get("task_type")) == "node"
    columns: dict[str, list[Any]] = {}
    for j, name in enumerate(output_columns(manifest)):
        columns[name] = _column(rows, j, node)
        if intervals is not None:
            columns[f"{name}_lower"] = _column(intervals["lower"], j, node)
            columns[f"{name}_upper"] = _column(intervals["upper"], j, node)
    if node:
        columns[ATOMIC_NUMBERS_COLUMN] = [list(z) for z in atomic_numbers or []]
    return pd.DataFrame(columns)


def _python_model_base() -> type:
    import mlflow.pyfunc

    return mlflow.pyfunc.PythonModel


class MiliaPackageModel(_python_model_base()):  # type: ignore[misc]
    """The pyfunc model of a deployment package (see the module docstring)."""

    def load_context(self, context: Any) -> None:
        from milia_pipeline.models.deployment.serving import PackagedPredictor

        self.packaged = PackagedPredictor.from_package(context.artifacts[PACKAGE_ARTIFACT])

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """One output row per input row (see the module docstring).

        Raises:
            ValueError: A row is neither a SMILES nor a structure, or cannot be parsed.
            InputRefusedError: Instances the model refuses, each named with its reason.
        """
        instances = instances_from_frame(model_input)
        rows, intervals, atomic_numbers = self.packaged.predict_response(instances)
        return predictions_frame(self.packaged.manifest, rows, intervals, atomic_numbers)
