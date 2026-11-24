"""Which inputs a trained model accepts, derived from what its checkpoint records.

A model is served only the inputs that reproduce its training graphs. Three kinds:

* ``identifier``: a molecule given by its molecular graph alone (SMILES, InChI, or a structure
  file whose coordinates the model does not use). MILIA featurizes it as training did
  (``structural_features_config``, hydrogens, pre-transforms).
* ``structure``: a molecule with given coordinates, built the way the training dataset built its
  molecules (``data_info.molecule_creation``). Not yet built by this MILIA: refused with the
  reason, and the model is predicted from a dataset.
* ``dataset``: graphs from a processed dataset of the same type, which carry every value the
  model was trained on. Always accepted.

``identifier`` is accepted only when the checkpoint records everything identifier featurization
must reproduce (``explicit_hydrogens``, ``qm_input_features``, ``input_transforms``), the model
takes no value from its dataset's QM data, and it does not read atomic positions. A checkpoint
written before a record existed accepts ``dataset`` only: its answer is unknown, and serving it
as if it were "no" would feed the model inputs unlike its training data (fail closed).
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
    "InputContract",
    "input_contract",
    "model_requires_positions",
]

IDENTIFIER = "identifier"
STRUCTURE = "structure"
DATASET = "dataset"
INPUT_KINDS = (IDENTIFIER, STRUCTURE, DATASET)

# data_info entries identifier featurization reproduces; a checkpoint without one cannot say
IDENTIFIER_RECORDS = ("explicit_hydrogens", "qm_input_features", "input_transforms")

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

    missing = [name for name in IDENTIFIER_RECORDS if name not in data_info]
    qm_features = list(data_info.get("qm_input_features") or [])
    if missing:
        refused[IDENTIFIER] = (
            f"the checkpoint does not record {', '.join(missing)} (written before MILIA recorded "
            "them), so identifier featurization cannot be shown to match training; retrain, or "
            f"{_DATASET_ADVICE}"
        )
    elif qm_features:
        refused[IDENTIFIER] = (
            f"the model takes {', '.join(qm_features)} from its dataset's QM data, which a "
            f"SMILES, InChI or structure file does not carry; {_DATASET_ADVICE}"
        )
    elif requires_positions:
        refused[IDENTIFIER] = (
            "the model reads atomic positions; an identifier has none, and MILIA does not "
            f"generate coordinates for a model trained on given ones; {_DATASET_ADVICE}"
        )

    refused[STRUCTURE] = (
        "this MILIA does not build structure inputs with the training dataset's molecule "
        f"creation yet; {_DATASET_ADVICE}"
    )
    return InputContract(refused)
