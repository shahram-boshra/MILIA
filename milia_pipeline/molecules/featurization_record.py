"""What inference must reproduce of the training graphs' molecular representation.

Records in ``data_info``: ``explicit_hydrogens`` (below); ``qm_input_features``, the input columns
whose training values came from the dataset's QM data (its atomic charges, bond lengths of the
QM-optimized geometry, per-atom properties added to ``x``), and ``qm_input_sources``, where each of
those values came from (``dataset_property``: a per-atom value only the dataset holds;
``coordinates``: the given geometry); ``molecule_creation``, how the dataset built each molecule.
No identifier (SMILES, InChI) carries QM values, so inference refuses identifiers for such a model
(``models/input_contract.py``): it is predicted from a dataset of the same type
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
    "SOURCE_COORDINATES",
    "SOURCE_DATASET_PROPERTY",
    "graphs_have_explicit_hydrogens",
    "molecular_charges",
    "molecule_creation_record",
    "qm_input_features",
    "qm_input_sources",
]

SOURCE_DATASET_PROPERTY = "dataset_property"  # a per-atom value only the dataset holds
SOURCE_COORDINATES = "coordinates"  # computed from the dataset's given geometry

SAMPLE_GRAPHS = 1000  # graphs inspected; a dataset either keeps hydrogens or does not


def qm_input_features(
    structural_features_config: dict[str, Any] | None,
    graphs: Iterable[Any] = (),
    node_features: Iterable[str] = (),
) -> list[str]:
    """The input columns a training run took from its dataset's QM data (``qm_input_sources``)."""
    return list(qm_input_sources(structural_features_config, graphs, node_features))


def qm_input_sources(
    structural_features_config: dict[str, Any] | None,
    graphs: Iterable[Any] = (),
    node_features: Iterable[str] = (),
) -> dict[str, str]:
    """The input columns a training run took from its dataset's QM data, and their source.

    Per-atom properties appended to ``x`` (``node_features``, the dataset's
    ``node_features_to_add``) are dataset values by definition and are recorded as
    ``x.<property>``.

    A feature's inputs come from the feature catalog (``StructuralFeature.reads`` in
    ``mol_structural_features``). Whether the dataset supplied them is decided as the molecule
    converter decides what it passes to the structural featurization: the dataset's atomic
    charges (``should_pass_atomic_charges_to_structural_features``: passing is on and the dataset
    configures an ``atomic_charges_property``) and its coordinates
    (``should_pass_coordinates_to_structural_features``), the latter only when the training
    graphs carry coordinates of their own (``pos``): a dataset without coordinates computed its
    geometric features from a generated conformer, as inference does.

    Returns:
        Catalog name (``atom.<feature>`` / ``bond.<feature>``; a deprecated configured name is
        recorded under its catalog name) → source, in configuration order, then
        ``x.<property>`` → ``dataset_property``. A feature reading the dataset's atomic charges
        has source ``dataset_property``; one reading only its coordinates, ``coordinates``. Empty
        when every input column can be computed from a molecule's identifier.
    """
    dataset_columns = {
        f"x.{name}": SOURCE_DATASET_PROPERTY for name in dict.fromkeys(node_features or ())
    }
    if not structural_features_config:
        return dataset_columns
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
        return dataset_columns

    sources: dict[str, str] = {}
    for level in ("atom", "bond"):
        for configured in structural_features_config.get(level) or []:
            name = DEPRECATED_FEATURE_NAMES[level].get(configured, configured)
            feature = get_structural_feature(level, name)
            reads = feature.reads & supplied if feature is not None else set()
            if reads and f"{level}.{name}" not in sources:
                sources[f"{level}.{name}"] = (
                    SOURCE_DATASET_PROPERTY if READS_ATOMIC_CHARGES in reads else SOURCE_COORDINATES
                )
    return {**sources, **dataset_columns}


def molecular_charges(graphs: Any) -> list[int] | None:
    """The molecular charges the training molecules were built with (sorted, distinct).

    Read from each graph's ``molecular_charge`` (set when the converter builds the molecule); a
    dataset stored as one collated object (``InMemoryDataset``), or a ``Subset`` of one (the
    training split), is read in one pass.

    Returns:
        The charges, or ``None`` when a graph does not carry one (processed before MILIA recorded
        it: the charges are unknown, so structure inputs are refused until re-processing).
    """
    import torch

    base, indices = graphs, None
    if isinstance(graphs, torch.utils.data.Subset):
        base, indices = graphs.dataset, list(graphs.indices)
    stored = getattr(base, "_data", None)
    if getattr(base, "_indices", None) is not None:  # a view: map through its own indices
        view = list(base.indices())
        indices = view if indices is None else [view[i] for i in indices]
    if stored is not None and "molecular_charge" in stored:
        charges = stored.molecular_charge.reshape(-1)
        if charges.numel() == base.len():  # one entry per stored graph (InMemoryDataset.len)
            selected = charges if indices is None else charges[torch.as_tensor(indices)]
            return sorted({int(c) for c in selected.tolist()})
    seen: set[int] = set()
    for graph in graphs:
        charge = getattr(graph, "molecular_charge", None)
        if charge is None:
            return None
        seen.add(int(charge.reshape(-1)[0]))
    return sorted(seen)


def molecule_creation_record(dataset_type: str | None, graphs: Any = None) -> dict[str, Any] | None:
    """How the training dataset built each molecule, from its registered dataset class.

    ``strategy``: ``identifier_coordinate_based`` (bonds from the identifier, the given
    coordinates assigned through the atom mapping) or ``coordinate_based`` (bonds determined from
    the given coordinates); ``identifier_keys``: the ``(key, type)`` pairs it reads in priority
    order; ``coordinate_units``: the units of its coordinates; ``charges``: the molecular charges
    its molecules were built with (:func:`molecular_charges` of ``graphs``; ``None`` when
    unknown). A structure input must be built the same way, with a charge among these.

    Returns:
        The record, or ``None`` when no dataset class is registered under ``dataset_type``.
    """
    if not dataset_type:
        return None
    from milia_pipeline.datasets import registry
    from milia_pipeline.exceptions import DatasetNotFoundError

    try:
        dataset_class = registry.get(dataset_type)
    except DatasetNotFoundError:
        return None
    return {
        "dataset_type": dataset_type,
        "strategy": dataset_class.get_molecule_creation_strategy(),
        "identifier_keys": [list(pair) for pair in dataset_class.get_identifier_keys()],
        "coordinate_units": dataset_class.get_coordinate_units(),
        "charges": molecular_charges(graphs) if graphs is not None else None,
    }


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
