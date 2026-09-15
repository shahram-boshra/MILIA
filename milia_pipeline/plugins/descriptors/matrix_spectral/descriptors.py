"""
Eigenvalue / matrix-spectral descriptors for MILIA (descriptor programme, Pace 5 / v1.8.0).

First-party PLUGIN-NATIVE descriptors implemented from primary literature and validated against
the ``mordredcommunity`` oracle (rel-err <= 1e-4; in practice <= 6e-8, the residual being
Floyd-Warshall / eigen summation order). Zero-core-modification plugin contract
(``name`` == ``function_name``; NaN on any failure or disconnected graph, never raise).

Families (177), all on the heavy-atom graph (2D), ``require_connected`` -> NaN when disconnected:
    AdjacencyMatrix   Sp*/VE*/VR*/LogEE (12)             adjacency-matrix spectra
    DistanceMatrix    Sp*/VE*/VR*/LogEE (12)             distance-matrix spectra
    DetourMatrix      + SM1 (13)                         detour-matrix spectra
    BaryszMatrix      Sp*/VE*/VR*/LogEE/SM1 x 8 props    Barysz Z-weighted distance spectra (104)
    BCUT              Burden eigenvalues x 12 props x hi/lo (24)   Pearlman-Smith BCUT
    MolecularId       MID/AMID x {any,hetero,C,N,O,X} (12)         Randic molecular ID

Spectral descriptors (Todeschini & Consonni): SpAbs=graph energy, SpMax=leading eigenvalue,
SpDiam, SpAD/SpMAD=(mean) absolute spectral deviation, LogEE=Estrada-like log-sum-exp, SM1=trace,
VE1-3=leading-eigenvector coefficient sums, VR1-3=Randic eigenvector-based indices.

Author: Asadollah (Shahram) Boshra
License: MIT
"""

from __future__ import annotations

import logging
import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

logger = logging.getLogger(__name__)
_NAN = float("nan")
_PT = Chem.GetPeriodicTable()
_HALOGEN = {9, 17, 35, 53, 85}

# --- primary-literature atomic-property tables (element -> value) ---
_MASS = {
    1: 1.008,
    2: 4.002602,
    3: 6.94,
    4: 9.012182,
    5: 10.81,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    9: 18.998403,
    10: 20.1797,
    11: 22.989769,
    12: 24.305,
    13: 26.981539,
    14: 28.085,
    15: 30.973762,
    16: 32.06,
    17: 35.45,
    18: 39.948,
    19: 39.0983,
    20: 40.078,
    21: 44.955912,
    22: 47.867,
    23: 50.9415,
    24: 51.9961,
    25: 54.938045,
    26: 55.845,
    27: 58.933195,
    28: 58.6934,
    29: 63.546,
    30: 65.38,
    31: 69.723,
    32: 72.63,
    33: 74.9216,
    34: 78.96,
    35: 79.904,
    36: 83.798,
    37: 85.4678,
    38: 87.62,
    39: 88.90585,
    40: 91.224,
    41: 92.90638,
    42: 95.96,
    43: 98.0,
    44: 101.07,
    45: 102.9055,
    46: 106.42,
    47: 107.8682,
    48: 112.411,
    49: 114.818,
    50: 118.71,
    51: 121.76,
    52: 127.6,
    53: 126.90447,
}
_VDWV = {
    1: 5.57528,
    2: 11.49404,
    3: 25.252407,
    4: 15.002475,
    5: 29.647788,
    6: 20.579526,
    7: 15.598531,
    8: 14.710227,
    9: 13.305788,
    10: 15.298568,
    11: 48.996627,
    12: 21.68837,
    13: 26.094085,
    14: 38.792386,
    15: 24.429024,
    16: 24.429024,
    17: 22.449298,
    18: 27.833137,
    19: 87.113746,
    20: 51.632666,
    21: 41.629768,
    22: 39.349206,
    23: 37.153493,
    24: 36.617633,
    25: 36.086951,
    26: 35.561421,
    27: 33.510322,
    28: 32.024864,
    29: 31.539647,
    30: 34.015494,
    31: 27.391349,
    32: 39.349206,
    33: 26.521849,
    34: 28.730912,
    35: 26.521849,
    36: 34.525718,
    37: 116.524298,
    38: 64.667586,
    39: 52.306127,
    40: 46.45187,
    41: 43.396838,
    42: 42.802369,
    43: 42.213354,
    44: 40.47878,
    45: 38.792386,
    46: 38.792386,
    47: 39.349206,
    48: 43.396838,
    49: 30.113452,
    50: 42.802369,
    51: 36.617633,
    52: 36.617633,
    53: 32.515032,
}
_SANDERSON = {
    1: 2.592,
    3: 0.67,
    4: 1.81,
    5: 2.275,
    6: 2.746,
    7: 3.194,
    8: 3.654,
    9: 4.0,
    10: 4.5,
    11: 0.56,
    12: 1.318,
    13: 1.714,
    14: 2.138,
    15: 2.515,
    16: 2.957,
    17: 3.475,
    18: 3.31,
    19: 0.445,
    20: 0.946,
    21: 1.02,
    22: 1.09,
    23: 1.39,
    24: 1.66,
    25: 2.2,
    26: 2.2,
    27: 2.56,
    28: 1.94,
    29: 2.033,
    30: 2.223,
    31: 2.419,
    32: 2.618,
    33: 2.816,
    34: 3.014,
    35: 3.219,
    36: 2.91,
    37: 0.312,
    38: 0.721,
    39: 0.65,
    40: 0.9,
    41: 1.42,
    42: 1.15,
    47: 1.826,
    48: 1.978,
    49: 2.138,
    50: 2.298,
    51: 2.458,
    52: 2.618,
    53: 2.778,
}
_PAULING = {
    1: 2.2,
    3: 0.98,
    4: 1.57,
    5: 2.04,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    11: 0.93,
    12: 1.31,
    13: 1.61,
    14: 1.9,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    19: 0.82,
    20: 1.0,
    21: 1.36,
    22: 1.54,
    23: 1.63,
    24: 1.66,
    25: 1.55,
    26: 1.83,
    27: 1.88,
    28: 1.91,
    29: 1.9,
    30: 1.65,
    31: 1.81,
    32: 2.01,
    33: 2.18,
    34: 2.55,
    35: 2.96,
    36: 3.0,
    37: 0.82,
    38: 0.95,
    39: 1.22,
    40: 1.33,
    41: 1.6,
    42: 2.16,
    43: 1.9,
    44: 2.2,
    45: 2.28,
    46: 2.2,
    47: 1.93,
    48: 1.69,
    49: 1.78,
    50: 1.96,
    51: 2.05,
    52: 2.1,
    53: 2.66,
}
_ALLRED_ROCHOW = {
    1: 2.2,
    3: 0.97,
    4: 1.47,
    5: 2.01,
    6: 2.5,
    7: 3.07,
    8: 3.5,
    9: 4.1,
    11: 1.01,
    12: 1.23,
    13: 1.47,
    14: 1.74,
    15: 2.06,
    16: 2.44,
    17: 2.83,
    19: 0.91,
    20: 1.04,
    21: 1.2,
    22: 1.32,
    23: 1.45,
    24: 1.56,
    25: 1.6,
    26: 1.64,
    27: 1.7,
    28: 1.75,
    29: 1.75,
    30: 1.66,
    31: 1.82,
    32: 2.02,
    33: 2.2,
    34: 2.48,
    35: 2.74,
    37: 0.89,
    38: 0.99,
    39: 1.11,
    40: 1.22,
    41: 1.23,
    42: 1.3,
    43: 1.36,
    44: 1.42,
    45: 1.45,
    46: 1.35,
    47: 1.42,
    48: 1.46,
    49: 1.49,
    50: 1.72,
    51: 1.82,
    52: 2.01,
    53: 2.21,
}
_POLARIZ = {
    1: 0.666793,
    2: 0.205052,
    3: 24.33,
    4: 5.6,
    5: 3.03,
    6: 1.67,
    7: 1.1,
    8: 0.802,
    9: 0.557,
    10: 0.39432,
    11: 24.11,
    12: 10.6,
    13: 6.8,
    14: 5.53,
    15: 3.63,
    16: 2.9,
    17: 2.18,
    18: 1.6411,
    19: 43.06,
    20: 22.8,
    21: 17.8,
    22: 14.6,
    23: 12.4,
    24: 11.6,
    25: 9.4,
    26: 8.4,
    27: 7.5,
    28: 6.8,
    29: 6.2,
    30: 5.75,
    31: 8.12,
    32: 5.84,
    33: 4.31,
    34: 3.77,
    35: 3.05,
    36: 2.4844,
    37: 47.24,
    38: 23.5,
    39: 22.7,
    40: 17.9,
    41: 15.7,
    42: 12.8,
    43: 11.4,
    44: 9.6,
    45: 8.6,
    46: 4.8,
    47: 6.78,
    48: 7.36,
    49: 10.2,
    50: 7.84,
    51: 6.6,
    52: 5.5,
    53: 5.35,
}
_IONIZ = {
    1: 13.598443,
    2: 24.587387,
    3: 5.391719,
    4: 9.3227,
    5: 8.29802,
    6: 11.2603,
    7: 14.5341,
    8: 13.61805,
    9: 17.4228,
    10: 21.56454,
    11: 5.139076,
    12: 7.646235,
    13: 5.985768,
    14: 8.15168,
    15: 10.48669,
    16: 10.36001,
    17: 12.96763,
    18: 15.75961,
    19: 4.340663,
    20: 6.11316,
    21: 6.56149,
    22: 6.82812,
    23: 6.74619,
    24: 6.76651,
    25: 7.43402,
    26: 7.9024,
    27: 7.88101,
    28: 7.6398,
    29: 7.72638,
    30: 9.394199,
    31: 5.999301,
    32: 7.89943,
    33: 9.7886,
    34: 9.75239,
    35: 11.8138,
    36: 13.99961,
    37: 4.177128,
    38: 5.69485,
    39: 6.2173,
    40: 6.6339,
    41: 6.75885,
    42: 7.09243,
    43: 7.28,
    44: 7.3605,
    45: 7.4589,
    46: 8.3369,
    47: 7.57623,
    48: 8.99382,
    49: 5.78636,
    50: 7.34392,
    51: 8.60839,
    52: 9.0096,
    53: 10.45126,
}
_PERIOD = {
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 2,
    6: 2,
    7: 2,
    8: 2,
    9: 2,
    10: 2,
    11: 3,
    12: 3,
    13: 3,
    14: 3,
    15: 3,
    16: 3,
    17: 3,
    18: 3,
    19: 4,
    20: 4,
    21: 4,
    22: 4,
    23: 4,
    24: 4,
    25: 4,
    26: 4,
    27: 4,
    28: 4,
    29: 4,
    30: 4,
    31: 4,
    32: 4,
    33: 4,
    34: 4,
    35: 4,
    36: 4,
    37: 5,
    38: 5,
    39: 5,
    40: 5,
    41: 5,
    42: 5,
    43: 5,
    44: 5,
    45: 5,
    46: 5,
    47: 5,
    48: 5,
    49: 5,
    50: 5,
    51: 5,
    52: 5,
    53: 5,
}
_PROP_TABLES = {
    "m": _MASS,
    "v": _VDWV,
    "se": _SANDERSON,
    "pe": _PAULING,
    "are": _ALLRED_ROCHOW,
    "p": _POLARIZ,
    "i": _IONIZ,
}


# --------------------------------------------------------------------------- graph helpers
def _connected(mol) -> bool:
    n = mol.GetNumAtoms()
    if n == 0:
        return False
    adj = [[] for _ in range(n)]
    for b in mol.GetBonds():
        adj[b.GetBeginAtomIdx()].append(b.GetEndAtomIdx())
        adj[b.GetEndAtomIdx()].append(b.GetBeginAtomIdx())
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


def _adjacency(mol):
    return Chem.GetAdjacencyMatrix(mol).astype(float)


def _distance(mol):
    return Chem.GetDistanceMatrix(mol).astype(float)


def _detour(mol):
    n = mol.GetNumAtoms()
    adj = [[] for _ in range(n)]
    for b in mol.GetBonds():
        adj[b.GetBeginAtomIdx()].append(b.GetEndAtomIdx())
        adj[b.GetEndAtomIdx()].append(b.GetBeginAtomIdx())
    import sys

    sys.setrecursionlimit(100000)

    def dfs(u, length, best, seen):
        if length > best.get(u, -1):
            best[u] = length
        seen[u] = True
        for w in adj[u]:
            if not seen[w]:
                dfs(w, length + 1, best, seen)
        seen[u] = False

    d = np.zeros((n, n))
    for s in range(n):
        best: dict = {}
        dfs(s, 0, best, [False] * n)
        for t, ln in best.items():
            d[s, t] = ln
    return d


# --------------------------------------------------------------------------- spectral engine
def _spectral(mol, matrix, kind):
    if matrix is None or not _connected(mol):
        return _NAN
    n = mol.GetNumAtoms()
    w, v = np.linalg.eigh(matrix)
    imin = int(np.argmin(w))
    imax = int(np.argmax(w))
    if kind == "SpAbs":
        return float(np.abs(w).sum())
    if kind == "SpMax":
        return float(w[imax])
    if kind == "SpDiam":
        return float(w[imax] - w[imin])
    if kind == "SpAD":
        return float(np.abs(w - np.mean(w)).sum())
    if kind == "SpMAD":
        return float(np.abs(w - np.mean(w)).sum() / n)
    if kind == "LogEE":
        a = max(w[imax], 0.0)
        return float(a + np.log(np.exp(w - a).sum() + np.exp(-a)))
    if kind == "SM1":
        return float(np.trace(matrix))
    if kind == "VE1":
        return float(np.abs(v[:, imax]).sum())
    if kind == "VE2":
        return float(np.abs(v[:, imax]).sum() / n)
    if kind == "VE3":
        val = 0.1 * n * np.abs(v[:, imax]).sum()
        return float(np.log(val)) if val > 0 else _NAN
    # VR1/VR2/VR3
    s = 0.0
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        s += (v[i, imax] * v[j, imax]) ** -0.5
    s = s.real if isinstance(s, complex) else s
    if kind == "VR1":
        return float(s)
    if kind == "VR2":
        return float(s / n)
    val = 0.1 * n * s
    return float(np.log(val)) if (not isinstance(val, complex) and val > 0) else _NAN


# --------------------------------------------------------------------------- Barysz
def _prop_value(short, z):
    if short == "Z":
        return float(z)
    return _PROP_TABLES[short].get(z)


def _barysz(mol, short):
    n = mol.GetNumAtoms()
    p = [_prop_value(short, a.GetAtomicNum()) for a in mol.GetAtoms()]
    if any(x is None or x == 0 for x in p):
        return None
    c = _prop_value(short, 6)
    m = np.full((n, n), math.inf)
    np.fill_diagonal(m, 0.0)
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        w = (c * c) / (p[i] * p[j] * b.GetBondTypeAsDouble())
        if w < m[i, j]:
            m[i, j] = w
            m[j, i] = w
    for k in range(n):
        m = np.minimum(m, m[:, k, None] + m[None, k, :])
    for i in range(n):
        m[i, i] = 1.0 - c / p[i]
    return m


# --------------------------------------------------------------------------- BCUT
def _sigma_electrons(a):
    return sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum() != 1)


def _valence_delta(a):
    nnum = a.GetAtomicNum()
    if nnum == 1:
        return 0.0
    zv = _PT.GetNOuterElecs(nnum) - a.GetFormalCharge()
    z = nnum - a.GetFormalCharge()
    h = a.GetTotalNumHs() + sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum() == 1)
    return (zv - h) / (z - zv - 1)


def _intrinsic_state(a):
    d = _sigma_electrons(a)
    if d == 0:
        return _NAN
    per = _PERIOD.get(a.GetAtomicNum())
    if not per:
        return _NAN
    return ((2.0 / per) ** 2 * _valence_delta(a) + 1) / d


def _bcut_prop(mol, short):
    if short == "Z":
        return [float(a.GetAtomicNum()) for a in mol.GetAtoms()]
    if short == "d":
        return [float(_sigma_electrons(a)) for a in mol.GetAtoms()]
    if short == "dv":
        return [_valence_delta(a) for a in mol.GetAtoms()]
    if short == "s":
        return [_intrinsic_state(a) for a in mol.GetAtoms()]
    if short == "c":
        mm = Chem.Mol(mol)
        rdPartialCharges.ComputeGasteigerCharges(mm)
        out = []
        for a in mm.GetAtoms():
            val = a.GetDoubleProp("_GasteigerCharge")
            if a.HasProp("_GasteigerHCharge"):
                val += a.GetDoubleProp("_GasteigerHCharge")
            out.append(val)
        return out
    return [_prop_value(short, a.GetAtomicNum()) for a in mol.GetAtoms()]


def _burden(mol):
    n = mol.GetNumAtoms()
    m = 0.001 * np.ones((n, n))
    for b in mol.GetBonds():
        a = b.GetBeginAtom()
        c = b.GetEndAtom()
        w = b.GetBondTypeAsDouble() / 10.0
        if a.GetDegree() == 1 or c.GetDegree() == 1:
            w += 0.01
        m[a.GetIdx(), c.GetIdx()] = w
        m[c.GetIdx(), a.GetIdx()] = w
    return m


def _bcut(mol, short, high):
    if not _connected(mol):
        return _NAN
    p = _bcut_prop(mol, short)
    if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in p):
        return _NAN
    b = _burden(mol).copy()
    np.fill_diagonal(b, p)
    ev = np.linalg.eig(b)[0]
    if np.iscomplexobj(ev):
        ev = ev.real
    ev = np.sort(ev)[::-1]
    return float(ev[0] if high else ev[-1])


# --------------------------------------------------------------------------- MolecularId
def _atomic_ids(mol, eps=1e-10):
    import sys

    sys.setrecursionlimit(100000)
    n = mol.GetNumAtoms()
    g = [[] for _ in range(n)]
    for b in mol.GetBonds():
        a = b.GetBeginAtom()
        c = b.GetEndAtom()
        w = a.GetDegree() * c.GetDegree()
        g[a.GetIdx()].append((c.GetIdx(), w))
        g[c.GetIdx()].append((a.GetIdx(), w))
    lim = int(1.0 / (eps**2))

    def get_id(start):
        acc = [0.0]
        visited = set()
        weights = [1]

        def search(u):
            visited.add(u)
            for v, d in g[u]:
                if v in visited:
                    continue
                visited.add(v)
                w = d * weights[-1]
                weights.append(w)
                acc[0] += 1.0 / math.sqrt(w)
                if w < lim:
                    search(v)
                visited.remove(v)
                weights.pop()

        search(start)
        return acc[0]

    return [1 + get_id(i) / 2.0 for i in range(n)]


_MID_CHECK = {
    "any": lambda a: True,
    "h": lambda a: a not in (1, 6),
    "C": lambda a: a == 6,
    "N": lambda a: a == 7,
    "O": lambda a: a == 8,
    "X": lambda a: a in _HALOGEN,
}


def _molecular_id(mol, typ, averaged):
    if not _connected(mol):
        return _NAN
    aids = _atomic_ids(mol)
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    chk = _MID_CHECK[typ]
    total = float(sum(aids[i] for i in range(len(aids)) if chk(atoms[i])))
    if averaged:
        return total / len(aids) if aids else _NAN
    return total


def _safe(thunk):
    try:
        v = float(thunk())
        return v if math.isfinite(v) or math.isnan(v) else _NAN
    except Exception:
        return _NAN


# --------------------------------------------------------------------------- descriptors
def SpAbs_A(mol):
    """SpAbs_A (AdjacencyMatrix)."""
    return _safe(lambda: _spectral(mol, _adjacency(mol), "SpAbs"))


def SpMax_A(mol):
    """SpMax_A (AdjacencyMatrix)."""
    return _safe(lambda: _spectral(mol, _adjacency(mol), "SpMax"))


def SpDiam_A(mol):
    """SpDiam_A (AdjacencyMatrix)."""
    return _safe(lambda: _spectral(mol, _adjacency(mol), "SpDiam"))


def SpAD_A(mol):
    """SpAD_A (AdjacencyMatrix)."""
    return _safe(lambda: _spectral(mol, _adjacency(mol), "SpAD"))


def SpMAD_A(mol):
    """SpMAD_A (AdjacencyMatrix)."""
    return _safe(lambda: _spectral(mol, _adjacency(mol), "SpMAD"))


def LogEE_A(mol):
    """LogEE_A (AdjacencyMatrix)."""
    return _safe(lambda: _spectral(mol, _adjacency(mol), "LogEE"))


def VE1_A(mol):
    """VE1_A (AdjacencyMatrix)."""
    return _safe(lambda: _spectral(mol, _adjacency(mol), "VE1"))


def VE2_A(mol):
    """VE2_A (AdjacencyMatrix)."""
    return _safe(lambda: _spectral(mol, _adjacency(mol), "VE2"))


def VE3_A(mol):
    """VE3_A (AdjacencyMatrix)."""
    return _safe(lambda: _spectral(mol, _adjacency(mol), "VE3"))


def VR1_A(mol):
    """VR1_A (AdjacencyMatrix)."""
    return _safe(lambda: _spectral(mol, _adjacency(mol), "VR1"))


def VR2_A(mol):
    """VR2_A (AdjacencyMatrix)."""
    return _safe(lambda: _spectral(mol, _adjacency(mol), "VR2"))


def VR3_A(mol):
    """VR3_A (AdjacencyMatrix)."""
    return _safe(lambda: _spectral(mol, _adjacency(mol), "VR3"))


def SpAbs_D(mol):
    """SpAbs_D (DistanceMatrix)."""
    return _safe(lambda: _spectral(mol, _distance(mol), "SpAbs"))


def SpMax_D(mol):
    """SpMax_D (DistanceMatrix)."""
    return _safe(lambda: _spectral(mol, _distance(mol), "SpMax"))


def SpDiam_D(mol):
    """SpDiam_D (DistanceMatrix)."""
    return _safe(lambda: _spectral(mol, _distance(mol), "SpDiam"))


def SpAD_D(mol):
    """SpAD_D (DistanceMatrix)."""
    return _safe(lambda: _spectral(mol, _distance(mol), "SpAD"))


def SpMAD_D(mol):
    """SpMAD_D (DistanceMatrix)."""
    return _safe(lambda: _spectral(mol, _distance(mol), "SpMAD"))


def LogEE_D(mol):
    """LogEE_D (DistanceMatrix)."""
    return _safe(lambda: _spectral(mol, _distance(mol), "LogEE"))


def VE1_D(mol):
    """VE1_D (DistanceMatrix)."""
    return _safe(lambda: _spectral(mol, _distance(mol), "VE1"))


def VE2_D(mol):
    """VE2_D (DistanceMatrix)."""
    return _safe(lambda: _spectral(mol, _distance(mol), "VE2"))


def VE3_D(mol):
    """VE3_D (DistanceMatrix)."""
    return _safe(lambda: _spectral(mol, _distance(mol), "VE3"))


def VR1_D(mol):
    """VR1_D (DistanceMatrix)."""
    return _safe(lambda: _spectral(mol, _distance(mol), "VR1"))


def VR2_D(mol):
    """VR2_D (DistanceMatrix)."""
    return _safe(lambda: _spectral(mol, _distance(mol), "VR2"))


def VR3_D(mol):
    """VR3_D (DistanceMatrix)."""
    return _safe(lambda: _spectral(mol, _distance(mol), "VR3"))


def SpAbs_Dt(mol):
    """SpAbs_Dt (DetourMatrix)."""
    return _safe(lambda: _spectral(mol, _detour(mol), "SpAbs"))


def SpMax_Dt(mol):
    """SpMax_Dt (DetourMatrix)."""
    return _safe(lambda: _spectral(mol, _detour(mol), "SpMax"))


def SpDiam_Dt(mol):
    """SpDiam_Dt (DetourMatrix)."""
    return _safe(lambda: _spectral(mol, _detour(mol), "SpDiam"))


def SpAD_Dt(mol):
    """SpAD_Dt (DetourMatrix)."""
    return _safe(lambda: _spectral(mol, _detour(mol), "SpAD"))


def SpMAD_Dt(mol):
    """SpMAD_Dt (DetourMatrix)."""
    return _safe(lambda: _spectral(mol, _detour(mol), "SpMAD"))


def LogEE_Dt(mol):
    """LogEE_Dt (DetourMatrix)."""
    return _safe(lambda: _spectral(mol, _detour(mol), "LogEE"))


def SM1_Dt(mol):
    """SM1_Dt (DetourMatrix)."""
    return _safe(lambda: _spectral(mol, _detour(mol), "SM1"))


def VE1_Dt(mol):
    """VE1_Dt (DetourMatrix)."""
    return _safe(lambda: _spectral(mol, _detour(mol), "VE1"))


def VE2_Dt(mol):
    """VE2_Dt (DetourMatrix)."""
    return _safe(lambda: _spectral(mol, _detour(mol), "VE2"))


def VE3_Dt(mol):
    """VE3_Dt (DetourMatrix)."""
    return _safe(lambda: _spectral(mol, _detour(mol), "VE3"))


def VR1_Dt(mol):
    """VR1_Dt (DetourMatrix)."""
    return _safe(lambda: _spectral(mol, _detour(mol), "VR1"))


def VR2_Dt(mol):
    """VR2_Dt (DetourMatrix)."""
    return _safe(lambda: _spectral(mol, _detour(mol), "VR2"))


def VR3_Dt(mol):
    """VR3_Dt (DetourMatrix)."""
    return _safe(lambda: _spectral(mol, _detour(mol), "VR3"))


def SpAbs_DzZ(mol):
    """SpAbs_DzZ (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "Z"), "SpAbs"))


def SpMax_DzZ(mol):
    """SpMax_DzZ (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "Z"), "SpMax"))


def SpDiam_DzZ(mol):
    """SpDiam_DzZ (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "Z"), "SpDiam"))


def SpAD_DzZ(mol):
    """SpAD_DzZ (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "Z"), "SpAD"))


def SpMAD_DzZ(mol):
    """SpMAD_DzZ (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "Z"), "SpMAD"))


def LogEE_DzZ(mol):
    """LogEE_DzZ (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "Z"), "LogEE"))


def SM1_DzZ(mol):
    """SM1_DzZ (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "Z"), "SM1"))


def VE1_DzZ(mol):
    """VE1_DzZ (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "Z"), "VE1"))


def VE2_DzZ(mol):
    """VE2_DzZ (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "Z"), "VE2"))


def VE3_DzZ(mol):
    """VE3_DzZ (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "Z"), "VE3"))


def VR1_DzZ(mol):
    """VR1_DzZ (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "Z"), "VR1"))


def VR2_DzZ(mol):
    """VR2_DzZ (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "Z"), "VR2"))


def VR3_DzZ(mol):
    """VR3_DzZ (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "Z"), "VR3"))


def SpAbs_Dzm(mol):
    """SpAbs_Dzm (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "m"), "SpAbs"))


def SpMax_Dzm(mol):
    """SpMax_Dzm (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "m"), "SpMax"))


def SpDiam_Dzm(mol):
    """SpDiam_Dzm (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "m"), "SpDiam"))


def SpAD_Dzm(mol):
    """SpAD_Dzm (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "m"), "SpAD"))


def SpMAD_Dzm(mol):
    """SpMAD_Dzm (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "m"), "SpMAD"))


def LogEE_Dzm(mol):
    """LogEE_Dzm (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "m"), "LogEE"))


def SM1_Dzm(mol):
    """SM1_Dzm (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "m"), "SM1"))


def VE1_Dzm(mol):
    """VE1_Dzm (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "m"), "VE1"))


def VE2_Dzm(mol):
    """VE2_Dzm (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "m"), "VE2"))


def VE3_Dzm(mol):
    """VE3_Dzm (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "m"), "VE3"))


def VR1_Dzm(mol):
    """VR1_Dzm (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "m"), "VR1"))


def VR2_Dzm(mol):
    """VR2_Dzm (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "m"), "VR2"))


def VR3_Dzm(mol):
    """VR3_Dzm (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "m"), "VR3"))


def SpAbs_Dzv(mol):
    """SpAbs_Dzv (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "v"), "SpAbs"))


def SpMax_Dzv(mol):
    """SpMax_Dzv (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "v"), "SpMax"))


def SpDiam_Dzv(mol):
    """SpDiam_Dzv (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "v"), "SpDiam"))


def SpAD_Dzv(mol):
    """SpAD_Dzv (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "v"), "SpAD"))


def SpMAD_Dzv(mol):
    """SpMAD_Dzv (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "v"), "SpMAD"))


def LogEE_Dzv(mol):
    """LogEE_Dzv (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "v"), "LogEE"))


def SM1_Dzv(mol):
    """SM1_Dzv (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "v"), "SM1"))


def VE1_Dzv(mol):
    """VE1_Dzv (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "v"), "VE1"))


def VE2_Dzv(mol):
    """VE2_Dzv (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "v"), "VE2"))


def VE3_Dzv(mol):
    """VE3_Dzv (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "v"), "VE3"))


def VR1_Dzv(mol):
    """VR1_Dzv (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "v"), "VR1"))


def VR2_Dzv(mol):
    """VR2_Dzv (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "v"), "VR2"))


def VR3_Dzv(mol):
    """VR3_Dzv (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "v"), "VR3"))


def SpAbs_Dzse(mol):
    """SpAbs_Dzse (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "se"), "SpAbs"))


def SpMax_Dzse(mol):
    """SpMax_Dzse (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "se"), "SpMax"))


def SpDiam_Dzse(mol):
    """SpDiam_Dzse (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "se"), "SpDiam"))


def SpAD_Dzse(mol):
    """SpAD_Dzse (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "se"), "SpAD"))


def SpMAD_Dzse(mol):
    """SpMAD_Dzse (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "se"), "SpMAD"))


def LogEE_Dzse(mol):
    """LogEE_Dzse (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "se"), "LogEE"))


def SM1_Dzse(mol):
    """SM1_Dzse (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "se"), "SM1"))


def VE1_Dzse(mol):
    """VE1_Dzse (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "se"), "VE1"))


def VE2_Dzse(mol):
    """VE2_Dzse (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "se"), "VE2"))


def VE3_Dzse(mol):
    """VE3_Dzse (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "se"), "VE3"))


def VR1_Dzse(mol):
    """VR1_Dzse (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "se"), "VR1"))


def VR2_Dzse(mol):
    """VR2_Dzse (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "se"), "VR2"))


def VR3_Dzse(mol):
    """VR3_Dzse (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "se"), "VR3"))


def SpAbs_Dzpe(mol):
    """SpAbs_Dzpe (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "pe"), "SpAbs"))


def SpMax_Dzpe(mol):
    """SpMax_Dzpe (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "pe"), "SpMax"))


def SpDiam_Dzpe(mol):
    """SpDiam_Dzpe (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "pe"), "SpDiam"))


def SpAD_Dzpe(mol):
    """SpAD_Dzpe (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "pe"), "SpAD"))


def SpMAD_Dzpe(mol):
    """SpMAD_Dzpe (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "pe"), "SpMAD"))


def LogEE_Dzpe(mol):
    """LogEE_Dzpe (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "pe"), "LogEE"))


def SM1_Dzpe(mol):
    """SM1_Dzpe (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "pe"), "SM1"))


def VE1_Dzpe(mol):
    """VE1_Dzpe (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "pe"), "VE1"))


def VE2_Dzpe(mol):
    """VE2_Dzpe (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "pe"), "VE2"))


def VE3_Dzpe(mol):
    """VE3_Dzpe (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "pe"), "VE3"))


def VR1_Dzpe(mol):
    """VR1_Dzpe (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "pe"), "VR1"))


def VR2_Dzpe(mol):
    """VR2_Dzpe (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "pe"), "VR2"))


def VR3_Dzpe(mol):
    """VR3_Dzpe (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "pe"), "VR3"))


def SpAbs_Dzare(mol):
    """SpAbs_Dzare (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "are"), "SpAbs"))


def SpMax_Dzare(mol):
    """SpMax_Dzare (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "are"), "SpMax"))


def SpDiam_Dzare(mol):
    """SpDiam_Dzare (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "are"), "SpDiam"))


def SpAD_Dzare(mol):
    """SpAD_Dzare (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "are"), "SpAD"))


def SpMAD_Dzare(mol):
    """SpMAD_Dzare (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "are"), "SpMAD"))


def LogEE_Dzare(mol):
    """LogEE_Dzare (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "are"), "LogEE"))


def SM1_Dzare(mol):
    """SM1_Dzare (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "are"), "SM1"))


def VE1_Dzare(mol):
    """VE1_Dzare (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "are"), "VE1"))


def VE2_Dzare(mol):
    """VE2_Dzare (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "are"), "VE2"))


def VE3_Dzare(mol):
    """VE3_Dzare (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "are"), "VE3"))


def VR1_Dzare(mol):
    """VR1_Dzare (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "are"), "VR1"))


def VR2_Dzare(mol):
    """VR2_Dzare (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "are"), "VR2"))


def VR3_Dzare(mol):
    """VR3_Dzare (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "are"), "VR3"))


def SpAbs_Dzp(mol):
    """SpAbs_Dzp (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "p"), "SpAbs"))


def SpMax_Dzp(mol):
    """SpMax_Dzp (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "p"), "SpMax"))


def SpDiam_Dzp(mol):
    """SpDiam_Dzp (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "p"), "SpDiam"))


def SpAD_Dzp(mol):
    """SpAD_Dzp (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "p"), "SpAD"))


def SpMAD_Dzp(mol):
    """SpMAD_Dzp (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "p"), "SpMAD"))


def LogEE_Dzp(mol):
    """LogEE_Dzp (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "p"), "LogEE"))


def SM1_Dzp(mol):
    """SM1_Dzp (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "p"), "SM1"))


def VE1_Dzp(mol):
    """VE1_Dzp (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "p"), "VE1"))


def VE2_Dzp(mol):
    """VE2_Dzp (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "p"), "VE2"))


def VE3_Dzp(mol):
    """VE3_Dzp (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "p"), "VE3"))


def VR1_Dzp(mol):
    """VR1_Dzp (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "p"), "VR1"))


def VR2_Dzp(mol):
    """VR2_Dzp (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "p"), "VR2"))


def VR3_Dzp(mol):
    """VR3_Dzp (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "p"), "VR3"))


def SpAbs_Dzi(mol):
    """SpAbs_Dzi (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "i"), "SpAbs"))


def SpMax_Dzi(mol):
    """SpMax_Dzi (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "i"), "SpMax"))


def SpDiam_Dzi(mol):
    """SpDiam_Dzi (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "i"), "SpDiam"))


def SpAD_Dzi(mol):
    """SpAD_Dzi (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "i"), "SpAD"))


def SpMAD_Dzi(mol):
    """SpMAD_Dzi (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "i"), "SpMAD"))


def LogEE_Dzi(mol):
    """LogEE_Dzi (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "i"), "LogEE"))


def SM1_Dzi(mol):
    """SM1_Dzi (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "i"), "SM1"))


def VE1_Dzi(mol):
    """VE1_Dzi (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "i"), "VE1"))


def VE2_Dzi(mol):
    """VE2_Dzi (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "i"), "VE2"))


def VE3_Dzi(mol):
    """VE3_Dzi (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "i"), "VE3"))


def VR1_Dzi(mol):
    """VR1_Dzi (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "i"), "VR1"))


def VR2_Dzi(mol):
    """VR2_Dzi (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "i"), "VR2"))


def VR3_Dzi(mol):
    """VR3_Dzi (BaryszMatrix)."""
    return _safe(lambda: _spectral(mol, _barysz(mol, "i"), "VR3"))


def BCUTc_1h(mol):
    """BCUTc_1h (BCUT); Mordred canonical name: BCUTc-1h."""
    return _safe(lambda: _bcut(mol, "c", True))


def BCUTc_1l(mol):
    """BCUTc_1l (BCUT); Mordred canonical name: BCUTc-1l."""
    return _safe(lambda: _bcut(mol, "c", False))


def BCUTdv_1h(mol):
    """BCUTdv_1h (BCUT); Mordred canonical name: BCUTdv-1h."""
    return _safe(lambda: _bcut(mol, "dv", True))


def BCUTdv_1l(mol):
    """BCUTdv_1l (BCUT); Mordred canonical name: BCUTdv-1l."""
    return _safe(lambda: _bcut(mol, "dv", False))


def BCUTd_1h(mol):
    """BCUTd_1h (BCUT); Mordred canonical name: BCUTd-1h."""
    return _safe(lambda: _bcut(mol, "d", True))


def BCUTd_1l(mol):
    """BCUTd_1l (BCUT); Mordred canonical name: BCUTd-1l."""
    return _safe(lambda: _bcut(mol, "d", False))


def BCUTs_1h(mol):
    """BCUTs_1h (BCUT); Mordred canonical name: BCUTs-1h."""
    return _safe(lambda: _bcut(mol, "s", True))


def BCUTs_1l(mol):
    """BCUTs_1l (BCUT); Mordred canonical name: BCUTs-1l."""
    return _safe(lambda: _bcut(mol, "s", False))


def BCUTZ_1h(mol):
    """BCUTZ_1h (BCUT); Mordred canonical name: BCUTZ-1h."""
    return _safe(lambda: _bcut(mol, "Z", True))


def BCUTZ_1l(mol):
    """BCUTZ_1l (BCUT); Mordred canonical name: BCUTZ-1l."""
    return _safe(lambda: _bcut(mol, "Z", False))


def BCUTm_1h(mol):
    """BCUTm_1h (BCUT); Mordred canonical name: BCUTm-1h."""
    return _safe(lambda: _bcut(mol, "m", True))


def BCUTm_1l(mol):
    """BCUTm_1l (BCUT); Mordred canonical name: BCUTm-1l."""
    return _safe(lambda: _bcut(mol, "m", False))


def BCUTv_1h(mol):
    """BCUTv_1h (BCUT); Mordred canonical name: BCUTv-1h."""
    return _safe(lambda: _bcut(mol, "v", True))


def BCUTv_1l(mol):
    """BCUTv_1l (BCUT); Mordred canonical name: BCUTv-1l."""
    return _safe(lambda: _bcut(mol, "v", False))


def BCUTse_1h(mol):
    """BCUTse_1h (BCUT); Mordred canonical name: BCUTse-1h."""
    return _safe(lambda: _bcut(mol, "se", True))


def BCUTse_1l(mol):
    """BCUTse_1l (BCUT); Mordred canonical name: BCUTse-1l."""
    return _safe(lambda: _bcut(mol, "se", False))


def BCUTpe_1h(mol):
    """BCUTpe_1h (BCUT); Mordred canonical name: BCUTpe-1h."""
    return _safe(lambda: _bcut(mol, "pe", True))


def BCUTpe_1l(mol):
    """BCUTpe_1l (BCUT); Mordred canonical name: BCUTpe-1l."""
    return _safe(lambda: _bcut(mol, "pe", False))


def BCUTare_1h(mol):
    """BCUTare_1h (BCUT); Mordred canonical name: BCUTare-1h."""
    return _safe(lambda: _bcut(mol, "are", True))


def BCUTare_1l(mol):
    """BCUTare_1l (BCUT); Mordred canonical name: BCUTare-1l."""
    return _safe(lambda: _bcut(mol, "are", False))


def BCUTp_1h(mol):
    """BCUTp_1h (BCUT); Mordred canonical name: BCUTp-1h."""
    return _safe(lambda: _bcut(mol, "p", True))


def BCUTp_1l(mol):
    """BCUTp_1l (BCUT); Mordred canonical name: BCUTp-1l."""
    return _safe(lambda: _bcut(mol, "p", False))


def BCUTi_1h(mol):
    """BCUTi_1h (BCUT); Mordred canonical name: BCUTi-1h."""
    return _safe(lambda: _bcut(mol, "i", True))


def BCUTi_1l(mol):
    """BCUTi_1l (BCUT); Mordred canonical name: BCUTi-1l."""
    return _safe(lambda: _bcut(mol, "i", False))


def MID(mol):
    """MID (MolecularId)."""
    return _safe(lambda: _molecular_id(mol, "any", False))


def AMID(mol):
    """AMID (MolecularId)."""
    return _safe(lambda: _molecular_id(mol, "any", True))


def MID_h(mol):
    """MID_h (MolecularId)."""
    return _safe(lambda: _molecular_id(mol, "h", False))


def AMID_h(mol):
    """AMID_h (MolecularId)."""
    return _safe(lambda: _molecular_id(mol, "h", True))


def MID_C(mol):
    """MID_C (MolecularId)."""
    return _safe(lambda: _molecular_id(mol, "C", False))


def AMID_C(mol):
    """AMID_C (MolecularId)."""
    return _safe(lambda: _molecular_id(mol, "C", True))


def MID_N(mol):
    """MID_N (MolecularId)."""
    return _safe(lambda: _molecular_id(mol, "N", False))


def AMID_N(mol):
    """AMID_N (MolecularId)."""
    return _safe(lambda: _molecular_id(mol, "N", True))


def MID_O(mol):
    """MID_O (MolecularId)."""
    return _safe(lambda: _molecular_id(mol, "O", False))


def AMID_O(mol):
    """AMID_O (MolecularId)."""
    return _safe(lambda: _molecular_id(mol, "O", True))


def MID_X(mol):
    """MID_X (MolecularId)."""
    return _safe(lambda: _molecular_id(mol, "X", False))


def AMID_X(mol):
    """AMID_X (MolecularId)."""
    return _safe(lambda: _molecular_id(mol, "X", True))


ALL_DESCRIPTOR_NAMES = (
    "SpAbs_A",
    "SpMax_A",
    "SpDiam_A",
    "SpAD_A",
    "SpMAD_A",
    "LogEE_A",
    "VE1_A",
    "VE2_A",
    "VE3_A",
    "VR1_A",
    "VR2_A",
    "VR3_A",
    "SpAbs_D",
    "SpMax_D",
    "SpDiam_D",
    "SpAD_D",
    "SpMAD_D",
    "LogEE_D",
    "VE1_D",
    "VE2_D",
    "VE3_D",
    "VR1_D",
    "VR2_D",
    "VR3_D",
    "SpAbs_Dt",
    "SpMax_Dt",
    "SpDiam_Dt",
    "SpAD_Dt",
    "SpMAD_Dt",
    "LogEE_Dt",
    "SM1_Dt",
    "VE1_Dt",
    "VE2_Dt",
    "VE3_Dt",
    "VR1_Dt",
    "VR2_Dt",
    "VR3_Dt",
    "SpAbs_DzZ",
    "SpMax_DzZ",
    "SpDiam_DzZ",
    "SpAD_DzZ",
    "SpMAD_DzZ",
    "LogEE_DzZ",
    "SM1_DzZ",
    "VE1_DzZ",
    "VE2_DzZ",
    "VE3_DzZ",
    "VR1_DzZ",
    "VR2_DzZ",
    "VR3_DzZ",
    "SpAbs_Dzm",
    "SpMax_Dzm",
    "SpDiam_Dzm",
    "SpAD_Dzm",
    "SpMAD_Dzm",
    "LogEE_Dzm",
    "SM1_Dzm",
    "VE1_Dzm",
    "VE2_Dzm",
    "VE3_Dzm",
    "VR1_Dzm",
    "VR2_Dzm",
    "VR3_Dzm",
    "SpAbs_Dzv",
    "SpMax_Dzv",
    "SpDiam_Dzv",
    "SpAD_Dzv",
    "SpMAD_Dzv",
    "LogEE_Dzv",
    "SM1_Dzv",
    "VE1_Dzv",
    "VE2_Dzv",
    "VE3_Dzv",
    "VR1_Dzv",
    "VR2_Dzv",
    "VR3_Dzv",
    "SpAbs_Dzse",
    "SpMax_Dzse",
    "SpDiam_Dzse",
    "SpAD_Dzse",
    "SpMAD_Dzse",
    "LogEE_Dzse",
    "SM1_Dzse",
    "VE1_Dzse",
    "VE2_Dzse",
    "VE3_Dzse",
    "VR1_Dzse",
    "VR2_Dzse",
    "VR3_Dzse",
    "SpAbs_Dzpe",
    "SpMax_Dzpe",
    "SpDiam_Dzpe",
    "SpAD_Dzpe",
    "SpMAD_Dzpe",
    "LogEE_Dzpe",
    "SM1_Dzpe",
    "VE1_Dzpe",
    "VE2_Dzpe",
    "VE3_Dzpe",
    "VR1_Dzpe",
    "VR2_Dzpe",
    "VR3_Dzpe",
    "SpAbs_Dzare",
    "SpMax_Dzare",
    "SpDiam_Dzare",
    "SpAD_Dzare",
    "SpMAD_Dzare",
    "LogEE_Dzare",
    "SM1_Dzare",
    "VE1_Dzare",
    "VE2_Dzare",
    "VE3_Dzare",
    "VR1_Dzare",
    "VR2_Dzare",
    "VR3_Dzare",
    "SpAbs_Dzp",
    "SpMax_Dzp",
    "SpDiam_Dzp",
    "SpAD_Dzp",
    "SpMAD_Dzp",
    "LogEE_Dzp",
    "SM1_Dzp",
    "VE1_Dzp",
    "VE2_Dzp",
    "VE3_Dzp",
    "VR1_Dzp",
    "VR2_Dzp",
    "VR3_Dzp",
    "SpAbs_Dzi",
    "SpMax_Dzi",
    "SpDiam_Dzi",
    "SpAD_Dzi",
    "SpMAD_Dzi",
    "LogEE_Dzi",
    "SM1_Dzi",
    "VE1_Dzi",
    "VE2_Dzi",
    "VE3_Dzi",
    "VR1_Dzi",
    "VR2_Dzi",
    "VR3_Dzi",
    "BCUTc_1h",
    "BCUTc_1l",
    "BCUTdv_1h",
    "BCUTdv_1l",
    "BCUTd_1h",
    "BCUTd_1l",
    "BCUTs_1h",
    "BCUTs_1l",
    "BCUTZ_1h",
    "BCUTZ_1l",
    "BCUTm_1h",
    "BCUTm_1l",
    "BCUTv_1h",
    "BCUTv_1l",
    "BCUTse_1h",
    "BCUTse_1l",
    "BCUTpe_1h",
    "BCUTpe_1l",
    "BCUTare_1h",
    "BCUTare_1l",
    "BCUTp_1h",
    "BCUTp_1l",
    "BCUTi_1h",
    "BCUTi_1l",
    "MID",
    "AMID",
    "MID_h",
    "AMID_h",
    "MID_C",
    "AMID_C",
    "MID_N",
    "AMID_N",
    "MID_O",
    "AMID_O",
    "MID_X",
    "AMID_X",
)
