"""What inference must reproduce of the training graphs' molecular representation.

Two records, both in ``data_info``: ``explicit_hydrogens`` (below) and ``qm_input_features``,
the structural features whose training values came from the dataset's QM data (Mulliken
charges, bond lengths of the QM-optimized geometry). No identifier (SMILES, InChI) carries those
values, so inference refuses to featurize identifiers for such a model
(``Predictor.prepare_inputs``): it is predicted from a dataset of the same type
(``milia --predict --test-path <dataset directory>``), whose processing keeps the QM geometry and
charges as they are.

The training graphs of a dataset built from QM data (DFT, QM9, ...) carry explicit hydrogens: the
molecule is read from its identifier with hydrogens and the QM-optimized coordinates are assigned
to every atom (``identifier_coordinate_based``). A molecule featurized at inference from SMILES or
InChI has implicit hydrogens unless they are added, so the model would see fewer atoms than it was
trained on. Training records whether its graphs hold hydrogens (``data_info.explicit_hydrogens``)
and inference featurizes with ``add_hydrogens`` accordingly (``Predictor.featurization_kwargs``).
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from typing import Any

__all__ = [
    "QM_ATOM_FEATURES",
    "QM_GEOMETRY_BOND_FEATURES",
    "SAMPLE_GRAPHS",
    "graphs_have_explicit_hydrogens",
    "qm_input_features",
]

# Structural features whose training values come from the dataset's QM data rather than from the
# molecule's identifier: Mulliken charges (a QM result; an identifier gives none, so the feature
# is 0.0 at inference) and bond lengths of the QM-optimized geometry (an identifier only gives a
# generated conformer, a different geometry).
QM_ATOM_FEATURES = ("mulliken_charge",)
QM_GEOMETRY_BOND_FEATURES = ("bond_length", "bond_length_binned")

SAMPLE_GRAPHS = 1000  # graphs inspected; a dataset either keeps hydrogens or does not


def qm_input_features(
    structural_features_config: dict[str, Any] | None, graphs: Iterable[Any] = ()
) -> list[str]:
    """The structural features a training run took from its dataset's QM data.

    Decided as the molecule converter decides what it passes to the structural featurization
    (``should_pass_mulliken_charges_to_structural_features``, which also checks that the dataset
    type supports Mulliken charges; ``should_pass_coordinates_to_structural_features``), and, for
    the geometry, only when the training graphs carry coordinates of their own (``pos``): a
    dataset without coordinates computed its bond lengths from a generated conformer, as
    inference does.

    Returns:
        The names (``atom.<feature>`` / ``bond.<feature>``); empty when every feature can be
        computed from a molecule's identifier.
    """
    if not structural_features_config:
        return []
    from milia_pipeline.config.config_accessors import (
        should_pass_coordinates_to_structural_features,
        should_pass_mulliken_charges_to_structural_features,
    )

    atom = set(structural_features_config.get("atom") or [])
    bond = set(structural_features_config.get("bond") or [])
    names = []
    if should_pass_mulliken_charges_to_structural_features():
        names += [f"atom.{name}" for name in QM_ATOM_FEATURES if name in atom]
    if should_pass_coordinates_to_structural_features() and any(
        getattr(graph, "pos", None) is not None for graph in islice(graphs, SAMPLE_GRAPHS)
    ):
        names += [f"bond.{name}" for name in QM_GEOMETRY_BOND_FEATURES if name in bond]
    return names


def graphs_have_explicit_hydrogens(graphs: Iterable[Any]) -> bool | None:
    """Whether the graphs carry hydrogen atoms as nodes.

    Reads the atomic numbers (``z``) of up to ``SAMPLE_GRAPHS`` graphs.

    Returns:
        ``True`` when any graph has a hydrogen node, ``False`` when graphs with atomic numbers
        have none, ``None`` when the graphs carry no atomic numbers (nothing to tell).
    """
    seen = False
    for graph in islice(graphs, SAMPLE_GRAPHS):
        z = getattr(graph, "z", None)
        if z is None:
            return None
        seen = True
        if bool((z == 1).any()):
            return True
    return False if seen else None
