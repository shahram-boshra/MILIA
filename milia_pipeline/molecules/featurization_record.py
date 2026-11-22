"""What inference must reproduce of the training graphs' molecular representation.

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

__all__ = ["SAMPLE_GRAPHS", "graphs_have_explicit_hydrogens"]

SAMPLE_GRAPHS = 1000  # graphs inspected; a dataset either keeps hydrogens or does not


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
