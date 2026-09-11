"""
Constitutional-ext + property/physicochemical descriptors for MILIA
(descriptor programme, Pace 2 / v1.5.0).

This first-party plugin implements ``PLUGIN-NATIVE`` descriptors from primary literature
(Blueprint §1 "native-from-primary-literature" default) under the standard
zero-core-modification plugin contract (``calculate_<n>(mol) -> float``, NaN on any
failure, never raise). Every value is validated against an oracle: ``mordredcommunity``
(BSD-3) for the 18 Mordred-defined descriptors, and the published rule definitions for the
two absorption filters (Blueprint §6). Native output equals the oracle to ``rtol<=1e-4``
(in practice bit-identical) across the validation panel.

Descriptors (20) and provenance
-------------------------------
constitutional (block):
    McGowanVolume       VMcGowan                                     Abraham & McGowan 1987  (10.1007/BF02311772)
    VdwVolumeABC        Vabc                                         Zhao, Abraham & Zissimos 2003 (10.1021/jo034808o)
    Polarizability      apol, bpol                                   Miller 1990 / atomic polarizability table
    CarbonTypes         C1SP1 C2SP1 C1SP2 C2SP2 C3SP2 C1SP3 C2SP3
                        C3SP3 C4SP3 HybRatio                         Todeschini & Consonni 2009; Yang et al. 2010 (HybRatio)
    Framework           fMF                                          Bemis & Murcko 1996 (10.1021/jm9602928)
    FragmentComplexity  fragCpx                                      Nilakantan et al. 2006 (10.1021/ci050521b)
drug_likeness (block):
    Lipinski            Lipinski, GhoseFilter                        Lipinski et al. 2001; Ghose et al. 1999
    DrugLikenessFilter  VeberFilter, EganFilter                      Veber et al. 2002 (10.1021/jm020017n);
                                                                     Egan et al. 2000 (10.1021/jm000292e)

Design rules (Blueprint §3, §4)
-------------------------------
* Pure: imports only RDKit + stdlib (no optional extra; ``requires_extra`` is null for all).
* Deterministic and 2D-only: ``requires_3d = false`` for every descriptor (no conformer,
  no RNG).
* NaN, never raise: any invalid molecule / undefined value yields ``float('nan')``.
* Boolean rule flags (Lipinski, GhoseFilter, VeberFilter, EganFilter) honour the scalar
  ``-> float`` contract by returning ``1.0`` / ``0.0``.
* Single-registration invariant: each function is named identically to its descriptor
  ``name`` (``function_name == name``), so the plugin undeclared-scan registers each exactly
  once (mirrors the Pace 1 ``addcore_3d`` convention).

Hydrogen convention (verified against the oracle)
-------------------------------------------------
McGowan, Vabc, apol, bpol and Framework operate on the **H-included** graph (mordred's
default ``explicit_hydrogens=True``); CarbonTypes and FragmentComplexity operate on the
**heavy-atom** graph. These conventions are reproduced exactly below.

Author: Asadollah (Shahram) Boshra
License: MIT
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem import Lipinski as _Lip

logger = logging.getLogger(__name__)


# =============================================================================
# PRIMARY-LITERATURE CONSTANT TABLES (element -> value)
# =============================================================================
# Values are the published atomic constants (identical to those tabulated by the
# mordredcommunity oracle, which sources them from the same primary papers).

# McGowan characteristic atomic volumes (cm^3 mol^-1), Abraham & McGowan 1987.
# Vx = sum(atomic contributions, H included) - 6.56 * (number of bonds).
_MCGOWAN_VOLUME: dict[int, float] = {
    1: 8.71,
    5: 18.31,
    6: 16.35,
    7: 14.39,
    8: 12.43,
    9: 10.47,
    14: 26.83,
    15: 24.87,
    16: 22.91,
    17: 20.95,
    33: 29.42,
    34: 27.81,
    35: 26.21,
    53: 34.54,
}
_MCGOWAN_BOND_DECREMENT = 6.56

# Static atomic polarizabilities (Angstrom^3). Miller 1990 / van de Waterbeemd tabulation.
_POLARIZABILITY: dict[int, float] = {
    1: 0.666793,
    5: 3.03,
    6: 1.67,
    7: 1.10,
    8: 0.802,
    9: 0.557,
    14: 5.53,
    15: 3.63,
    16: 2.90,
    17: 2.18,
    33: 4.31,
    34: 3.77,
    35: 3.05,
    53: 5.35,
}

# Bondi van der Waals radii (Angstrom); atomic volume contribution = 4/3 * pi * r^3.
# Zhao, Abraham & Zissimos 2003 atom-and-bond-contribution vdW volume.
_BONDI_RADIUS: dict[int, float] = {
    1: 1.20,
    6: 1.70,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    17: 1.75,
    35: 1.85,
    15: 1.80,
    16: 1.80,
    33: 1.85,
    5: 2.13,
    14: 2.10,
    34: 1.90,
}
_VABC_ATOM_VOLUME: dict[int, float] = {
    z: 4.0 / 3.0 * math.pi * r**3 for z, r in _BONDI_RADIUS.items()
}
_VABC_BOND_DECREMENT = 5.92
_VABC_AROMATIC_RING_DECREMENT = 14.7
_VABC_ALIPHATIC_RING_DECREMENT = 3.8

# CarbonTypes hybridization map (RDKit HybridizationType -> SP order 1/2/3).
_HYBRIDIZATION: dict[object, int] = {
    Chem.HybridizationType.SP: 1,
    Chem.HybridizationType.SP2: 2,
    Chem.HybridizationType.SP3: 3,
    Chem.HybridizationType.SP3D: 3,
    Chem.HybridizationType.SP3D2: 3,
}

# Absorption-filter thresholds (published; no automated oracle).
_VEBER_MAX_ROT_BONDS = 10
_VEBER_MAX_TPSA = 140.0
_EGAN_MAX_TPSA = 131.6
_EGAN_MAX_WLOGP = 5.88


# =============================================================================
# HELPERS
# =============================================================================


def _finite_or_nan(value: float) -> float:
    """Coerce to float; map non-finite (NaN/Inf) to NaN for a uniform invalid sentinel."""
    v = float(value)
    return v if math.isfinite(v) else float("nan")


def _with_explicit_h(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy with explicit hydrogens (mordred's default graph)."""
    return Chem.AddHs(Chem.Mol(mol))


def _all_atoms_parameterized(mol: Chem.Mol, table: dict[int, float]) -> bool:
    """True iff every atom's element has a parameter in ``table`` (else skip to NaN)."""
    return all(atom.GetAtomicNum() in table for atom in mol.GetAtoms())


def _carbon_type_counts(mol: Chem.Mol) -> dict[int, dict[int, int]]:
    """
    CarbonTypes cache: counts[SP][nCarbon] over the heavy-atom, kekulized graph.

    For each carbon, ``SP`` is the hybridization order (1/2/3) and ``nCarbon`` is the
    number of carbon neighbours. Kekulization mirrors the oracle (aromatic carbons stay
    SP2); hydrogens are implicit.
    """
    work = Chem.Mol(mol)
    Chem.Kekulize(work, clearAromaticFlags=False)
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for atom in work.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        n_carbon = sum(nb.GetAtomicNum() == 6 for nb in atom.GetNeighbors())
        sp = _HYBRIDIZATION.get(atom.GetHybridization())
        counts[sp][n_carbon] += 1
    return counts


def _bfs_shortest_path(adjacency: dict, source, target) -> list | None:
    """Unweighted shortest path (node list) between two graph nodes, or None."""
    if source == target:
        return [source]
    predecessor = {source: None}
    queue: deque = deque([source])
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node]:
            if neighbour not in predecessor:
                predecessor[neighbour] = node
                if neighbour == target:
                    path = [neighbour]
                    while predecessor[path[-1]] is not None:
                        path.append(predecessor[path[-1]])
                    return path[::-1]
                queue.append(neighbour)
    return None


# =============================================================================
# CONSTITUTIONAL — volume / polarizability
# =============================================================================


def VMcGowan(mol: Chem.Mol) -> float:
    """McGowan characteristic volume (Abraham & McGowan 1987), NaN on failure."""
    try:
        if mol is None:
            return float("nan")
        work = _with_explicit_h(mol)
        if not _all_atoms_parameterized(work, _MCGOWAN_VOLUME):
            return float("nan")
        total = sum(_MCGOWAN_VOLUME[a.GetAtomicNum()] for a in work.GetAtoms())
        return _finite_or_nan(total - work.GetNumBonds() * _MCGOWAN_BOND_DECREMENT)
    except Exception:
        return float("nan")


def Vabc(mol: Chem.Mol) -> float:
    """van der Waals volume, atom-and-bond-contribution (Zhao-Abraham-Zissimos 2003)."""
    try:
        if mol is None:
            return float("nan")
        work = _with_explicit_h(mol)
        if not _all_atoms_parameterized(work, _VABC_ATOM_VOLUME):
            return float("nan")
        atom_volume = sum(_VABC_ATOM_VOLUME[a.GetAtomicNum()] for a in work.GetAtoms())
        n_bonds = work.GetNumBonds()
        ring_info = work.GetRingInfo()
        n_aromatic = 0
        n_aliphatic = 0
        for bond_ring in ring_info.BondRings():
            if all(work.GetBondWithIdx(bi).GetIsAromatic() for bi in bond_ring):
                n_aromatic += 1
            else:
                n_aliphatic += 1
        value = (
            atom_volume
            - _VABC_BOND_DECREMENT * n_bonds
            - _VABC_AROMATIC_RING_DECREMENT * n_aromatic
            - _VABC_ALIPHATIC_RING_DECREMENT * n_aliphatic
        )
        return _finite_or_nan(value)
    except Exception:
        return float("nan")


def apol(mol: Chem.Mol) -> float:
    """Sum of atomic polarizabilities over all atoms (H included)."""
    try:
        if mol is None:
            return float("nan")
        work = _with_explicit_h(mol)
        if not _all_atoms_parameterized(work, _POLARIZABILITY):
            return float("nan")
        return _finite_or_nan(sum(_POLARIZABILITY[a.GetAtomicNum()] for a in work.GetAtoms()))
    except Exception:
        return float("nan")


def bpol(mol: Chem.Mol) -> float:
    """Sum over bonds of the absolute atomic-polarizability difference (H included)."""
    try:
        if mol is None:
            return float("nan")
        work = _with_explicit_h(mol)
        if not _all_atoms_parameterized(work, _POLARIZABILITY):
            return float("nan")
        total = 0.0
        for bond in work.GetBonds():
            a = _POLARIZABILITY[bond.GetBeginAtom().GetAtomicNum()]
            b = _POLARIZABILITY[bond.GetEndAtom().GetAtomicNum()]
            total += abs(a - b)
        return _finite_or_nan(total)
    except Exception:
        return float("nan")


# =============================================================================
# CONSTITUTIONAL — CarbonTypes (11 -> we ship 10; FCSP3 == HAVE FractionCSP3, dropped)
# =============================================================================


def _carbon_type(mol: Chem.Mol, n_carbon: int, sp: int) -> float:
    try:
        if mol is None:
            return float("nan")
        return float(_carbon_type_counts(mol)[sp][n_carbon])
    except Exception:
        return float("nan")


def C1SP1(mol: Chem.Mol) -> float:
    """Number of SP carbons bound to 1 other carbon."""
    return _carbon_type(mol, 1, 1)


def C2SP1(mol: Chem.Mol) -> float:
    """Number of SP carbons bound to 2 other carbons."""
    return _carbon_type(mol, 2, 1)


def C1SP2(mol: Chem.Mol) -> float:
    """Number of SP2 carbons bound to 1 other carbon."""
    return _carbon_type(mol, 1, 2)


def C2SP2(mol: Chem.Mol) -> float:
    """Number of SP2 carbons bound to 2 other carbons."""
    return _carbon_type(mol, 2, 2)


def C3SP2(mol: Chem.Mol) -> float:
    """Number of SP2 carbons bound to 3 other carbons."""
    return _carbon_type(mol, 3, 2)


def C1SP3(mol: Chem.Mol) -> float:
    """Number of SP3 carbons bound to 1 other carbon."""
    return _carbon_type(mol, 1, 3)


def C2SP3(mol: Chem.Mol) -> float:
    """Number of SP3 carbons bound to 2 other carbons."""
    return _carbon_type(mol, 2, 3)


def C3SP3(mol: Chem.Mol) -> float:
    """Number of SP3 carbons bound to 3 other carbons."""
    return _carbon_type(mol, 3, 3)


def C4SP3(mol: Chem.Mol) -> float:
    """Number of SP3 carbons bound to 4 other carbons."""
    return _carbon_type(mol, 4, 3)


def HybRatio(mol: Chem.Mol) -> float:
    """Hybridization ratio N(SP3) / (N(SP2) + N(SP3)); NaN when both are 0 (Yang 2010)."""
    try:
        if mol is None:
            return float("nan")
        counts = _carbon_type_counts(mol)
        n_sp3 = sum(counts[3].values())
        n_sp2 = sum(counts[2].values())
        if n_sp3 == 0 and n_sp2 == 0:
            return float("nan")
        return _finite_or_nan(n_sp3 / (n_sp2 + n_sp3))
    except Exception:
        return float("nan")


# =============================================================================
# CONSTITUTIONAL — Framework / FragmentComplexity
# =============================================================================


def fMF(mol: Chem.Mol) -> float:
    """
    Bemis-Murcko molecular framework ratio: N(framework atoms) / N(all atoms, H incl).

    Framework atoms = all ring atoms plus the linker atoms lying on a shortest path
    between two ring systems (Bemis & Murcko 1996). Implemented in stdlib only.
    """
    try:
        if mol is None:
            return float("nan")
        ring_info = mol.GetRingInfo()
        rings = [tuple(r) for r in ring_info.AtomRings()]
        ring_node_of: dict[int, tuple] = {}
        for ring_index, ring in enumerate(rings):
            for atom_idx in ring:
                ring_node_of[atom_idx] = ("R", ring_index)  # later rings overwrite
        ring_nodes = list(set(ring_node_of.values()))

        adjacency: dict = defaultdict(set)
        for bond in mol.GetBonds():
            a = ring_node_of.get(bond.GetBeginAtomIdx(), ("A", bond.GetBeginAtomIdx()))
            b = ring_node_of.get(bond.GetEndAtomIdx(), ("A", bond.GetEndAtomIdx()))
            if a != b:
                adjacency[a].add(b)
                adjacency[b].add(a)

        linkers: set[int] = set()
        n_rings = len(ring_nodes)
        for i in range(n_rings):
            for j in range(i + 1, n_rings):
                path = _bfs_shortest_path(adjacency, ring_nodes[i], ring_nodes[j])
                if path:
                    linkers.update(idx for tag, idx in path if tag == "A")

        n_framework = len(linkers) + len({i for ring in rings for i in ring})
        n_total = _with_explicit_h(mol).GetNumAtoms()
        if n_total == 0:
            return float("nan")
        return _finite_or_nan(n_framework / n_total)
    except Exception:
        return float("nan")


def fragCpx(mol: Chem.Mol) -> float:
    """
    Fragment complexity |B^2 - A^2 + A| + H/100 (Nilakantan et al. 2006), heavy-atom graph.

    A = number of atoms, B = number of bonds, H = number of heteroatoms.
    """
    try:
        if mol is None:
            return float("nan")
        a = mol.GetNumAtoms()
        b = mol.GetNumBonds()
        h = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 6)
        return _finite_or_nan(abs(b**2 - a**2 + a) + h / 100.0)
    except Exception:
        return float("nan")


# =============================================================================
# DRUG_LIKENESS — rule filters (boolean -> 1.0 / 0.0)
# =============================================================================


def Lipinski(mol: Chem.Mol) -> float:
    """Lipinski rule of five (Lipinski et al. 2001): 1.0 if all criteria pass else 0.0."""
    try:
        if mol is None:
            return float("nan")
        passes = (
            _Lip.NumHDonors(mol) <= 5
            and _Lip.NumHAcceptors(mol) <= 10
            and Descriptors.MolWt(mol) <= 500
            and Crippen.MolLogP(mol) <= 5
        )
        return 1.0 if passes else 0.0
    except Exception:
        return float("nan")


def GhoseFilter(mol: Chem.Mol) -> float:
    """Ghose filter (Ghose et al. 1999): 1.0 if all criteria pass else 0.0."""
    try:
        if mol is None:
            return float("nan")
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        mr = Crippen.MolMR(mol)
        n_atoms = _with_explicit_h(mol).GetNumAtoms()
        passes = (
            160 <= mw <= 480 and 20 <= n_atoms <= 70 and -0.4 <= logp <= 5.6 and 40 <= mr <= 130
        )
        return 1.0 if passes else 0.0
    except Exception:
        return float("nan")


def VeberFilter(mol: Chem.Mol) -> float:
    """
    Veber oral-bioavailability filter (Veber et al. 2002): rotatable bonds <= 10 AND
    TPSA <= 140 Angstrom^2. Returns 1.0 if both pass else 0.0.
    """
    try:
        if mol is None:
            return float("nan")
        n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        passes = n_rot <= _VEBER_MAX_ROT_BONDS and tpsa <= _VEBER_MAX_TPSA
        return 1.0 if passes else 0.0
    except Exception:
        return float("nan")


def EganFilter(mol: Chem.Mol) -> float:
    """
    Egan absorption filter (Egan et al. 2000): TPSA <= 131.6 Angstrom^2 AND WLOGP <= 5.88.

    WLOGP is the Wildman-Crippen LogP (RDKit ``Crippen.MolLogP``), the open equivalent of
    the AlogP98 axis of the Egan "egg". Returns 1.0 if both pass else 0.0.
    """
    try:
        if mol is None:
            return float("nan")
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        wlogp = Crippen.MolLogP(mol)
        passes = tpsa <= _EGAN_MAX_TPSA and wlogp <= _EGAN_MAX_WLOGP
        return 1.0 if passes else 0.0
    except Exception:
        return float("nan")


# Public registry of every descriptor this module provides (tests/introspection).
ALL_DESCRIPTOR_NAMES: tuple[str, ...] = (
    "VMcGowan",
    "Vabc",
    "apol",
    "bpol",
    "C1SP1",
    "C2SP1",
    "C1SP2",
    "C2SP2",
    "C3SP2",
    "C1SP3",
    "C2SP3",
    "C3SP3",
    "C4SP3",
    "HybRatio",
    "fMF",
    "fragCpx",
    "Lipinski",
    "GhoseFilter",
    "VeberFilter",
    "EganFilter",
)
