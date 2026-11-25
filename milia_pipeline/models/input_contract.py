"""Which inputs a trained model accepts, derived from what its checkpoint records.

A model is served only the inputs that reproduce its training graphs. Three kinds:

* ``identifier``: a molecule given by its molecular graph alone (SMILES, InChI, or a structure
  file whose coordinates the model does not use). MILIA featurizes it as training did
  (``structural_features_config``, hydrogens, pre-transforms).
* ``structure``: a molecule with given coordinates (an XYZ, SDF or MOL file), built the way the
  training dataset built its molecules (``data_info.molecule_creation``,
  ``molecules/structure_inputs.py``) and through the training pre-transforms.
* ``dataset``: graphs from a processed dataset of the same type, which carry every value the
  model was trained on. Always accepted.

``identifier`` is accepted only when the checkpoint records everything identifier featurization
must reproduce (``explicit_hydrogens``, ``qm_input_features``, ``input_transforms``), the model
takes no value from its dataset's QM data, and it does not read atomic positions. A checkpoint
written before a record existed accepts ``dataset`` only: its answer is unknown, and serving it
as if it were "no" would feed the model inputs unlike its training data (fail closed).

``structure`` is accepted only when the checkpoint records how its molecules were built
(``molecule_creation`` with the charges seen and the converter switches), its featurization and
pre-transforms (``structural_features_config``, ``input_transforms``) and where every QM input came
from (``qm_input_sources``), and no input came from a per-atom value only the dataset holds
(``dataset_property``: a structure carries coordinates and a charge, not the dataset's QM data).
A model reading atomic positions accepts structures: they carry the positions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from milia_pipeline.exceptions import ConfigurationError

__all__ = [
    "IDENTIFIER",
    "STRUCTURE",
    "DATASET",
    "INPUT_KINDS",
    "IDENTIFIER_RECORDS",
    "STRUCTURE_RECORDS",
    "InputContract",
    "input_contract",
    "model_requires_positions",
]

IDENTIFIER = "identifier"
STRUCTURE = "structure"
DATASET = "dataset"
INPUT_KINDS = (IDENTIFIER, STRUCTURE, DATASET)

# data_info entries identifier featurization reproduces; a checkpoint without one cannot say
IDENTIFIER_RECORDS = (
    "explicit_hydrogens",
    "qm_input_features",
    "input_transforms",
    "input_filters",
)
# data_info entries a structure input is built from; a checkpoint without one cannot say
STRUCTURE_RECORDS = (
    "structural_features_config",
    "input_transforms",
    "qm_input_sources",
    "molecule_creation",
    "input_filters",
)
# Records whose writer always gives a value: ``None`` means the run did not record it (a
# checkpoint writer stores ``None`` for a record its training path did not compute), never
# "nothing". ``input_transforms`` ``None`` is "no pre-transform", ``explicit_hydrogens`` ``None``
# "no atomic numbers", ``structural_features_config`` ``None`` "no features".
_NONE_IS_UNRECORDED = (
    "qm_input_features",
    "qm_input_sources",
    "molecule_creation",
    "input_filters",
)


def _unrecorded(data_info: Mapping[str, Any], names: tuple[str, ...]) -> list[str]:
    return [
        name
        for name in names
        if name not in data_info or (name in _NONE_IS_UNRECORDED and data_info[name] is None)
    ]


_DATASET_ADVICE = (
    "predict from a dataset of the same type (milia --predict --test-path <dataset directory>)"
)


@dataclass(frozen=True)
class InputContract:
    """The input kinds a model accepts; every other kind with the reason it is refused."""

    refused: Mapping[str, str] = field(default_factory=dict)

    @property
    def accepted(self) -> list[str]:
        return [kind for kind in INPUT_KINDS if kind not in self.refused]

    def accepts(self, kind: str) -> bool:
        return kind in INPUT_KINDS and kind not in self.refused

    def require(self, kind: str) -> None:
        """Refuse an input kind the model does not accept.

        Raises:
            ConfigurationError: The reason, and the kinds the model accepts.
        """
        if kind not in INPUT_KINDS:
            raise ConfigurationError(f"Unknown input kind {kind!r}; known: {list(INPUT_KINDS)}")
        if kind in self.refused:
            raise ConfigurationError(
                f"This model does not accept {kind} inputs: {self.refused[kind]}. "
                f"Accepted: {', '.join(self.accepted)}."
            )

    def to_dict(self) -> dict[str, Any]:
        """``{"accepted": [...], "refused": {kind: reason}}`` (JSON, for package signatures)."""
        return {"accepted": self.accepted, "refused": dict(self.refused)}


def model_requires_positions(model: torch.nn.Module) -> bool:
    """Whether any network inside ``model`` reads atomic positions (``z``, ``pos``).

    Read from each submodule's attached capabilities (set by the factory for the hyperparameters
    it was built with) or, for a module without them, from its class's forward signature; a
    composed model (ensemble, wrapper) requires positions when one of its networks does.
    """
    from milia_pipeline.models.capabilities import attached_capabilities, requires_positions

    for module in model.modules():
        capabilities = attached_capabilities(module)
        if capabilities is not None:
            if capabilities.requires_positions:
                return True
        elif requires_positions(type(module)):
            return True
    return False


def input_contract(
    data_info: Mapping[str, Any] | None, *, requires_positions: bool
) -> InputContract:
    """The contract of a model with this ``data_info`` (checkpoint) and positional need."""
    data_info = data_info or {}
    refused: dict[str, str] = {}

    missing = _unrecorded(data_info, IDENTIFIER_RECORDS)
    qm_features = list(data_info.get("qm_input_features") or [])
    if missing:
        refused[IDENTIFIER] = (
            f"the checkpoint does not record {', '.join(missing)} (written before MILIA recorded "
            "them), so identifier featurization cannot be shown to match training; retrain, or "
            f"{_DATASET_ADVICE}"
        )
    elif qm_features:
        sources = data_info.get("qm_input_sources") or {}
        geometric = [name for name in qm_features if sources.get(name) == "coordinates"]
        dataset_only = [name for name in qm_features if name not in geometric]
        if dataset_only:
            refused[IDENTIFIER] = (
                f"the model takes {', '.join(dataset_only)} from its dataset's QM data, which a "
                f"SMILES or InChI does not carry; {_DATASET_ADVICE}"
            )
        else:
            refused[IDENTIFIER] = (
                f"the model takes {', '.join(geometric)} from the given coordinates, which a "
                "SMILES or InChI does not carry; give the molecule as a structure (XYZ, SDF, MOL "
                f"with its coordinates) when the model accepts structures, or {_DATASET_ADVICE}"
            )
    elif requires_positions:
        refused[IDENTIFIER] = (
            "the model reads atomic positions; an identifier has none, and MILIA does not "
            f"generate coordinates for a model trained on given ones; {_DATASET_ADVICE}"
        )

    reason = _structure_refusal(data_info)
    if reason is not None:
        refused[STRUCTURE] = reason
    return InputContract(refused)


def _structure_refusal(data_info: Mapping[str, Any]) -> str | None:
    """Why the model cannot take structure inputs, or ``None`` when it can."""
    from milia_pipeline.molecules.featurization_record import SOURCE_DATASET_PROPERTY

    missing = _unrecorded(data_info, STRUCTURE_RECORDS)
    creation = data_info.get("molecule_creation") or {}
    if missing:
        names = ", ".join(missing)
        return (
            f"the checkpoint does not record {names} (written before MILIA recorded them), so a "
            f"structure cannot be built as its training molecules were; retrain, or {_DATASET_ADVICE}"
        )
    unrecorded = [name for name in ("charges", "featurization") if creation.get(name) is None]
    if unrecorded:
        return (
            f"the checkpoint does not record molecule_creation.{', '.join(unrecorded)} (its "
            "dataset was processed, or the model trained, before MILIA recorded them); re-process "
            f"the dataset and retrain, or {_DATASET_ADVICE}"
        )
    dataset_inputs = [
        name
        for name, source in (data_info.get("qm_input_sources") or {}).items()
        if source == SOURCE_DATASET_PROPERTY
    ]
    if dataset_inputs:
        return (
            f"the model takes {', '.join(dataset_inputs)} from its dataset's QM data, which a "
            f"structure does not carry; {_DATASET_ADVICE}"
        )
    return None
