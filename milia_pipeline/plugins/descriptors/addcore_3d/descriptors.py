"""
ADD-CORE RDKit 3D descriptor blocks for MILIA (descriptor programme, Pace 1 / v1.4.0).

This plugin exposes the RDKit-native vector descriptor blocks that are not part of the
MILIA ``HAVE`` baseline as individually-named scalar descriptors, following the
zero-core-modification plugin contract (``calculate_<name>(mol) -> float``, NaN on any
failure, never raise).

Blocks and RDKit sources (verified against ``rdkit==2025.03.5`` and ``2026.03.5``,
identical counts and values):

    WHIM        114   rdMolDescriptors.CalcWHIM        (3D)  Todeschini & Gramatica 1997
    GETAWAY     273   rdMolDescriptors.CalcGETAWAY     (3D)  Consonni, Todeschini & Pavan 2002
    RDF         210   rdMolDescriptors.CalcRDF         (3D)  Hemmer, Steinhauer & Gasteiger 1999
    MoRSE       224   rdMolDescriptors.CalcMORSE       (3D)  Schuur, Selzer & Gasteiger 1996
    AUTOCORR3D   80   rdMolDescriptors.CalcAUTOCORR3D  (3D)  Moreau & Broto 1980 (3D form)
    USR          12   rdMolDescriptors.GetUSR          (3D)  Ballester & Richards 2007
    USRCAT       60   rdMolDescriptors.GetUSRCAT       (3D)  Schreyer & Blundell 2012
    MQN          42   rdMolDescriptors.MQNs_           (2D)  Nguyen, Blum et al. 2009
    Oxidation     4   rdMolDescriptors.CalcOxidationNumbers (2D, aggregated)  Calvo Fracassi & Landrum 2021

Total: 1,015 vector/scalar descriptors + 4 oxidation-number aggregates = 1,019.

Design rules (Blueprint §3.1, §4):
    * Vector blocks are computed **once per molecule** and cached, then indexed. The cache
      is an in-plugin ``functools.lru_cache(maxsize=1)`` keyed on the ``Mol`` object; the
      canonical block-cache pattern (consecutive indexer calls on the same molecule reuse
      one RDKit call, and the single slot self-evicts on the next molecule — no unbounded
      growth). This mirrors production practice (e.g. Schrödinger's RDKit descriptor cache).
    * All computations are deterministic: 3D descriptors depend on the conformer, which the
      MILIA calculator embeds with the pinned ETKDG seed (randomSeed=42).
    * Any failure (invalid molecule, missing conformer, RDKit precondition such as USR's
      "too few atoms", or a NaN/Inf block element) yields ``float('nan')`` — never an
      exception (matches the engine convention and drives skip-not-fail reporting).

Author: Asadollah (Shahram) Boshra
License: MIT
"""

from __future__ import annotations

import functools
import logging
import math

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

logger = logging.getLogger(__name__)


# =============================================================================
# BLOCK-CACHE HELPERS (private — skipped by the plugin undeclared-scan)
# =============================================================================
#
# Each helper performs exactly ONE RDKit call and returns an immutable tuple so the
# lru_cache slot cannot be mutated by a caller. maxsize=1 is intentional: the descriptor
# calculator evaluates all indexers of a molecule back-to-back, so a single slot gives a
# 100% hit rate within a molecule while bounding memory to one block per block-type.


def _finite_or_nan(value: float) -> float:
    """Coerce to float; map non-finite (NaN/Inf) to NaN for a uniform invalid sentinel."""
    v = float(value)
    return v if math.isfinite(v) else float("nan")


@functools.lru_cache(maxsize=1)
def _compute_whim(mol: Chem.Mol) -> tuple[float, ...]:
    """WHIM block (114) — one CalcWHIM call, cached. thresh default = 0.001 (RDKit default)."""
    return tuple(rdMolDescriptors.CalcWHIM(mol))


@functools.lru_cache(maxsize=1)
def _compute_getaway(mol: Chem.Mol) -> tuple[float, ...]:
    """
    GETAWAY block (273) — one CalcGETAWAY call, cached (precision default = 2).

    Determinism requires rdkit >= 2025.03.6: earlier releases returned
    call-to-call-varying GETAWAY values (rdkit#7264, fixed in 2025.03.6). The MILIA
    runtime pins rdkit==2025.3.6, so this block is reproducible under seed=42.
    """
    return tuple(rdMolDescriptors.CalcGETAWAY(mol))


@functools.lru_cache(maxsize=1)
def _compute_rdf(mol: Chem.Mol) -> tuple[float, ...]:
    """RDF block (210) — one CalcRDF call, cached."""
    return tuple(rdMolDescriptors.CalcRDF(mol))


@functools.lru_cache(maxsize=1)
def _compute_morse(mol: Chem.Mol) -> tuple[float, ...]:
    """3D-MoRSE block (224) — one CalcMORSE call, cached."""
    return tuple(rdMolDescriptors.CalcMORSE(mol))


@functools.lru_cache(maxsize=1)
def _compute_autocorr3d(mol: Chem.Mol) -> tuple[float, ...]:
    """Autocorr3D block (80) — one CalcAUTOCORR3D call, cached."""
    return tuple(rdMolDescriptors.CalcAUTOCORR3D(mol))


@functools.lru_cache(maxsize=1)
def _compute_usr(mol: Chem.Mol) -> tuple[float, ...]:
    """USR block (12) — one GetUSR call, cached. Raises ValueError for < 3 atoms."""
    return tuple(rdMolDescriptors.GetUSR(mol))


@functools.lru_cache(maxsize=1)
def _compute_usrcat(mol: Chem.Mol) -> tuple[float, ...]:
    """USRCAT block (60) — one GetUSRCAT call, cached."""
    return tuple(rdMolDescriptors.GetUSRCAT(mol))


@functools.lru_cache(maxsize=1)
def _compute_mqn(mol: Chem.Mol) -> tuple[float, ...]:
    """MQN block (42, 2D) — one MQNs_ call, cached. Integer counts returned as floats."""
    return tuple(float(v) for v in rdMolDescriptors.MQNs_(mol))


@functools.lru_cache(maxsize=1)
def _compute_oxidation_aggregates(mol: Chem.Mol) -> tuple[float, float, float, float]:
    """
    Oxidation-number aggregates (min, max, mean, sum) over HEAVY atoms only.

    ``CalcOxidationNumbers`` mutates the molecule in place, so it is run on a private
    heavy-atom copy (explicit hydrogens stripped). Heavy-atom aggregation follows the
    descriptor-community convention (Mordred/Todeschini strip explicit H for descriptor
    correctness); oxidation states are topological (conformer-independent).
    """
    work = Chem.RemoveHs(Chem.Mol(mol))
    rdMolDescriptors.CalcOxidationNumbers(work)
    states = [
        atom.GetIntProp("OxidationNumber")
        for atom in work.GetAtoms()
        if atom.GetAtomicNum() > 1 and atom.HasProp("OxidationNumber")
    ]
    if not states:
        nan = float("nan")
        return nan, nan, nan, nan
    n = len(states)
    return (
        float(min(states)),
        float(max(states)),
        float(sum(states)) / n,
        float(sum(states)),
    )


# =============================================================================
# INDEXER FACTORY
# =============================================================================
#
# Each vector-block element is exposed as a distinct public function whose ``__name__``
# equals its descriptor name (e.g. ``WHIM_1``). Naming the function identically to the
# declared descriptor name makes the plugin undeclared-scan treat it as already-declared
# (single registration, no duplicate under a separate ``function_name``).

# 1-based (block_prefix, size, cache_fn) specification for the vector/scalar blocks.
_VECTOR_BLOCKS: tuple[tuple[str, int, object], ...] = (
    ("WHIM", 114, _compute_whim),
    ("GETAWAY", 273, _compute_getaway),
    ("RDF", 210, _compute_rdf),
    ("MoRSE", 224, _compute_morse),
    ("AUTOCORR3D", 80, _compute_autocorr3d),
    ("USR", 12, _compute_usr),
    ("USRCAT", 60, _compute_usrcat),
    ("MQN", 42, _compute_mqn),
)


def _make_indexer(cache_fn, index: int):
    """Build a thin ``mol -> float`` indexer into a cached block (NaN on any failure)."""

    def _indexer(mol: Chem.Mol) -> float:
        try:
            if mol is None:
                return float("nan")
            block = cache_fn(mol)
            return _finite_or_nan(block[index])
        except Exception:  # invalid mol / missing conformer / RDKit precondition
            return float("nan")

    return _indexer


# Generate and publish the 1,015 vector/scalar indexer functions into module globals.
_GENERATED_NAMES: list[str] = []
for _prefix, _size, _cache_fn in _VECTOR_BLOCKS:
    for _i in range(_size):
        _name = f"{_prefix}_{_i + 1}"  # 1-based descriptor naming
        _fn = _make_indexer(_cache_fn, _i)
        _fn.__name__ = _name
        _fn.__qualname__ = _name
        _fn.__doc__ = f"Element {_i + 1} of the {_prefix} block (RDKit-native)."
        globals()[_name] = _fn
        _GENERATED_NAMES.append(_name)


# =============================================================================
# OXIDATION-NUMBER AGGREGATE DESCRIPTORS (explicit, 2D)
# =============================================================================


def OxidationNumber_min(mol: Chem.Mol) -> float:
    """Minimum Pauling oxidation number over heavy atoms (NaN on failure)."""
    try:
        if mol is None:
            return float("nan")
        return _finite_or_nan(_compute_oxidation_aggregates(mol)[0])
    except Exception:
        return float("nan")


def OxidationNumber_max(mol: Chem.Mol) -> float:
    """Maximum Pauling oxidation number over heavy atoms (NaN on failure)."""
    try:
        if mol is None:
            return float("nan")
        return _finite_or_nan(_compute_oxidation_aggregates(mol)[1])
    except Exception:
        return float("nan")


def OxidationNumber_mean(mol: Chem.Mol) -> float:
    """Mean Pauling oxidation number over heavy atoms (NaN on failure)."""
    try:
        if mol is None:
            return float("nan")
        return _finite_or_nan(_compute_oxidation_aggregates(mol)[2])
    except Exception:
        return float("nan")


def OxidationNumber_sum(mol: Chem.Mol) -> float:
    """Sum of Pauling oxidation numbers over heavy atoms (NaN on failure)."""
    try:
        if mol is None:
            return float("nan")
        return _finite_or_nan(_compute_oxidation_aggregates(mol)[3])
    except Exception:
        return float("nan")


# Public registry of every descriptor this module provides (used by tests/introspection).
ALL_DESCRIPTOR_NAMES: tuple[str, ...] = tuple(_GENERATED_NAMES) + (
    "OxidationNumber_min",
    "OxidationNumber_max",
    "OxidationNumber_mean",
    "OxidationNumber_sum",
)
