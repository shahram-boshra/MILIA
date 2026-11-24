"""``data_info`` records a training run writes, for test checkpoints saved directly.

A checkpoint without them accepts dataset inputs only (``models/input_contract.py``: a record
that is absent cannot be shown to match training). Tests that serve, package or featurize
identifier inputs save their checkpoints with :data:`IDENTIFIER_DATA_INFO`: the records of a run
on heavy-atom graphs featurized from identifiers, with no QM input and no pre-transform.
"""

IDENTIFIER_DATA_INFO = {
    "explicit_hydrogens": False,
    "qm_input_features": [],
    "qm_input_sources": {},
    "input_transforms": None,
    "molecule_creation": None,
}
