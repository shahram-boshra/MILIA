"""What inference must reproduce of the training graphs' molecular representation.

Two records, both in ``data_info``: ``explicit_hydrogens`` (below) and ``qm_input_features``,
the structural features whose training values came from the dataset's QM data (its atomic
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
    "SAMPLE_GRAPHS",
    "graphs_have_explicit_hydrogens",
    "qm_input_features",
]

SAMPLE_GRAPHS = 1000  # graphs inspected; a dataset either keeps hydrogens or does not


def qm_input_features(
    structural_features_config: dict[str, Any] | None, graphs: Iterable[Any] = ()
) -> list[str]:
    """The structural features a training run took from its dataset's QM data.

    A feature's inputs come from the feature catalog (``StructuralFeature.reads`` in
    ``mol_structural_features``). Whether the dataset supplied them is decided as the molecule
    converter decides what it passes to the structural featurization: the dataset's atomic
    charges (``should_pass_atomic_charges_to_structural_features``: passing is on and the dataset
    configures an ``atomic_charges_property``) and its coordinates
    (``should_pass_coordinates_to_structural_features``), the latter only when the training
    graphs carry coordinates of their own (``pos``): a dataset without coordinates computed its
    geometric features from a generated conformer, as inference does.

    Returns:
        The catalog names (``atom.<feature>`` / ``bond.<feature>``; a deprecated configured name
        is recorded under its catalog name), in configuration order; empty when every feature
        can be computed from a molecule's identifier.
    """
    if not structural_features_config:
        return []
    from milia_pipeline.config.config_accessors import (
        should_pass_atomic_charges_to_structural_features,
        should_pass_coordinates_to_structural_features,
    )
    from milia_pipeline.molecules.mol_structural_features import (
        DEPRECATED_FEATURE_NAMES,
        READS_ATOMIC_CHARGES,
        READS_COORDINATES,
        get_structural_feature,
    )

    supplied = set()
    if should_pass_atomic_charges_to_structural_features():
        supplied.add(READS_ATOMIC_CHARGES)
    if should_pass_coordinates_to_structural_features() and any(
        getattr(graph, "pos", None) is not None for graph in islice(graphs, SAMPLE_GRAPHS)
    ):
        supplied.add(READS_COORDINATES)
    if not supplied:
        return []

    names: list[str] = []
    for level in ("atom", "bond"):
        for configured in structural_features_config.get(level) or []:
            name = DEPRECATED_FEATURE_NAMES[level].get(configured, configured)
            feature = get_structural_feature(level, name)
            if feature is None or not feature.reads & supplied:
                continue
            if f"{level}.{name}" not in names:
                names.append(f"{level}.{name}")
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
