"""
Walk/path + detour + information-content descriptors for MILIA
(descriptor programme, Pace 4 / v1.7.0).

First-party PLUGIN-NATIVE descriptors implemented from primary literature and validated
bit-exact against the ``mordredcommunity`` oracle (rel-err <= 1e-4). Standard
zero-core-modification plugin contract (``name`` == ``function_name``; NaN on any failure,
never raise).

Families (86)
-------------
topological (block):
    WalkCount                MWC01-10, TMWC10, SRW02-10, TSRW10   Rücker & Rücker 1993; Harary 1969
    PathCount                MPC2-10, TMPC10, piPC1-10, TpiPC10   Randić 1979
    DetourIndex              DetourIndex                          Trinajstić, Chemical Graph Theory
    InformationContent       IC/TIC/SIC/BIC/CIC/MIC/ZMIC x 0-5    Bonchev & Trinajstić 1977; Basak 1999
    VertexAdjacencyInformation VAdjMat                            Todeschini & Consonni, Handbook

Conventions (verified against the oracle)
-----------------------------------------
* WalkCount / PathCount / DetourIndex / VAdjMat operate on the heavy-atom graph.
* InformationContent operates on the **H-included, kekulized** graph (the oracle's default).
* MWC01 = number of edges; walk/path counts of order >= 2 use the ``log(x + 1)`` transform.
* ``MWC03 = 2 * Zagreb2`` is a known affine correlation (distinct value), not a duplicate.

Author: Asadollah (Shahram) Boshra
License: MIT
"""

from __future__ import annotations

import logging
import math
from itertools import chain, groupby

import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)
_NAN = float("nan")
_DETOUR_STEP_BUDGET = 3_000_000  # bound the NP-hard longest-simple-path DFS; else skip to NaN


class _BudgetExceeded(Exception):
    pass


# --------------------------------------------------------------------------- WalkCount
def _adj(mol):
    return Chem.GetAdjacencyMatrix(mol).astype(float)


def _mwc(mol, k):
    a = _adj(mol)
    if k == 1:
        return 0.5 * a.sum()
    return math.log(np.linalg.matrix_power(a, k).sum() + 1)


def _srw(mol, k):
    a = _adj(mol)
    return math.log(np.trace(np.linalg.matrix_power(a, k)) + 1)


def _tmwc10(mol):
    return float(mol.GetNumAtoms() + sum(_mwc(mol, k) for k in range(1, 11)))


def _tsrw10(mol):
    return float(mol.GetNumAtoms() + sum(_srw(mol, k) for k in range(1, 11)))


# --------------------------------------------------------------------------- VertexAdjacencyInformation
def _vadjmat(mol):
    m = sum(
        1
        for b in mol.GetBonds()
        if b.GetBeginAtom().GetAtomicNum() != 1 and b.GetEndAtom().GetAtomicNum() != 1
    )
    if m <= 0:
        return _NAN
    return 1.0 + math.log2(m)


# --------------------------------------------------------------------------- PathCount
def _path_count(mol, order):
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]

    def bond_ids_to_atom_ids(p):
        it = iter(p)
        try:
            a0f, a0t = bonds[next(it)]
        except StopIteration:
            return []
        try:
            a1f, a1t = bonds[next(it)]
        except StopIteration:
            return [a0f, a0t]
        if a0f in (a1f, a1t):
            path = [a0t, a0f]
            cur = a1f if a0f == a1t else a1t
        else:
            path = [a0f, a0t]
            cur = a1f if a0t == a1t else a1t
        for i in it:
            anf, ant = bonds[i]
            path.append(cur)
            cur = ant if anf == cur else anf
        path.append(cur)
        return path

    n_paths = 0
    pi = 0.0
    for path in Chem.FindAllPathsOfLengthN(mol, order):
        aids = set()
        before = None
        w = 1.0
        ok = True
        for i in bond_ids_to_atom_ids(path):
            if i in aids:
                ok = False
                break
            aids.add(i)
            if before is not None:
                w *= mol.GetBondBetweenAtoms(before, i).GetBondTypeAsDouble()
            before = i
        if ok:
            n_paths += 1
            pi += w
    return n_paths, pi


def _mpc(mol, k):
    return float(_path_count(mol, k)[0])


def _pipc(mol, k):
    return math.log(_path_count(mol, k)[1] + 1)


def _tmpc10(mol):
    return float(mol.GetNumAtoms() + sum(_path_count(mol, k)[0] for k in range(1, 11)))


def _tpipc10(mol):
    return math.log(mol.GetNumAtoms() + sum(_path_count(mol, k)[1] for k in range(1, 11)) + 1)


# --------------------------------------------------------------------------- DetourIndex
def _detour_index(mol):
    n = mol.GetNumAtoms()
    if n < 2:
        return 0.0
    adj = [[] for _ in range(n)]
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[u].append(v)
        adj[v].append(u)
    import sys

    sys.setrecursionlimit(100000)

    def dfs(u, length, best, seen, steps):
        steps[0] += 1
        if steps[0] > _DETOUR_STEP_BUDGET:
            raise _BudgetExceeded
        if length > best.get(u, -1):
            best[u] = length
        seen[u] = True
        for w in adj[u]:
            if not seen[w]:
                dfs(w, length + 1, best, seen, steps)
        seen[u] = False

    dmat = np.zeros((n, n))
    for s in range(n):
        best: dict = {}
        seen = [False] * n
        steps = [0]
        dfs(s, 0, best, seen, steps)
        for t, ln in best.items():
            dmat[s, t] = ln
    return float(int(0.5 * dmat.sum()))


# --------------------------------------------------------------------------- InformationContent
class _BFSTree:
    __slots__ = ("tree", "visited", "bonds", "atoms")

    def __init__(self, mol):
        self.tree = {}
        self.visited = set()
        self.bonds = {}
        for b in mol.GetBonds():
            s = b.GetBeginAtomIdx()
            d = b.GetEndAtomIdx()
            t = b.GetBondType()
            self.bonds[s, d] = t
            self.bonds[d, s] = t
        self.atoms = [(a.GetAtomicNum(), a.GetDegree(), a.GetNeighbors()) for a in mol.GetAtoms()]

    def reset(self, i):
        self.tree = {i: ()}
        self.visited = {i}

    def expand(self):
        self._expand(self.tree)

    def _expand(self, tree):
        for src, dst in list(tree.items()):
            self.visited.add(src)
            if not dst:
                tree[src] = {
                    n.GetIdx(): () for n in self.atoms[src][2] if n.GetIdx() not in self.visited
                }
            else:
                self._expand(dst)

    def _code(self, tree, before, trail):
        if len(tree) == 0:
            yield trail
        else:
            for src, dst in tree.items():
                code = []
                if before is not None:
                    code.append(self.bonds[before, src])
                code.append(self.atoms[src][:2])
                nxt = tuple(chain(trail, code))
                yield from self._code(dst, src, nxt)

    def get_code(self, i, order):
        self.reset(i)
        for _ in range(order):
            self.expand()
        return tuple(sorted(self._code(self.tree, None, ())))


def _ic_mol(mol):
    m = Chem.AddHs(Chem.Mol(mol))
    Chem.Kekulize(m, clearAromaticFlags=True)
    return m


def _ags(km, order):
    if order == 0:
        atoms = [a.GetAtomicNum() for a in km.GetAtoms()]
    else:
        tree = _BFSTree(km)
        atoms = [tree.get_code(i, order) for i in range(km.GetNumAtoms())]
    ad = {a: i for i, a in enumerate(atoms)}
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(sorted(atoms))]
    ids = np.fromiter((ad[k] for k, _ in groups), "int", len(groups))
    sizes = np.fromiter((g for _, g in groups), "float", len(groups))
    return ids, sizes


def _shannon(a, w=1):
    total = np.sum(a)
    p = a / total
    return float(-np.sum(w * (p * np.log2(p))))


def _ic(mol, m):
    _, a = _ags(_ic_mol(mol), m)
    return _shannon(a)


def _tic(mol, m):
    km = _ic_mol(mol)
    _, a = _ags(km, m)
    return float(km.GetNumAtoms() * _shannon(a))


def _sic(mol, m):
    km = _ic_mol(mol)
    _, a = _ags(km, m)
    d = math.log2(km.GetNumAtoms()) if km.GetNumAtoms() > 0 else 0
    return _shannon(a) / d if d else _NAN


def _bic(mol, m):
    km = _ic_mol(mol)
    _, a = _ags(km, m)
    b = sum(bd.GetBondTypeAsDouble() for bd in km.GetBonds())
    if b <= 0:
        return _NAN
    lb = math.log2(b)
    return _shannon(a) / lb if lb else _NAN


def _cic(mol, m):
    km = _ic_mol(mol)
    _, a = _ags(km, m)
    return float(math.log2(km.GetNumAtoms()) - _shannon(a))


def _mic(mol, m):
    km = _ic_mol(mol)
    ids, a = _ags(km, m)
    w = np.array([km.GetAtomWithIdx(int(i)).GetMass() for i in ids])
    return _shannon(a, w)


def _zmic(mol, m):
    km = _ic_mol(mol)
    ids, a = _ags(km, m)
    w = a * np.array([km.GetAtomWithIdx(int(i)).GetAtomicNum() for i in ids])
    return _shannon(a, w)


def _safe(fn, *args):
    try:
        v = float(fn(*args))
        return v if math.isfinite(v) or math.isnan(v) else _NAN
    except Exception:
        return _NAN


# --------------------------------------------------------------------------- descriptors
def MWC01(mol):
    """MWC01 (WalkCount)."""
    return _safe(_mwc, mol, 1)


def MWC02(mol):
    """MWC02 (WalkCount)."""
    return _safe(_mwc, mol, 2)


def MWC03(mol):
    """MWC03 (WalkCount)."""
    return _safe(_mwc, mol, 3)


def MWC04(mol):
    """MWC04 (WalkCount)."""
    return _safe(_mwc, mol, 4)


def MWC05(mol):
    """MWC05 (WalkCount)."""
    return _safe(_mwc, mol, 5)


def MWC06(mol):
    """MWC06 (WalkCount)."""
    return _safe(_mwc, mol, 6)


def MWC07(mol):
    """MWC07 (WalkCount)."""
    return _safe(_mwc, mol, 7)


def MWC08(mol):
    """MWC08 (WalkCount)."""
    return _safe(_mwc, mol, 8)


def MWC09(mol):
    """MWC09 (WalkCount)."""
    return _safe(_mwc, mol, 9)


def MWC10(mol):
    """MWC10 (WalkCount)."""
    return _safe(_mwc, mol, 10)


def TMWC10(mol):
    """TMWC10 (WalkCount)."""
    return _safe(_tmwc10, mol)


def SRW02(mol):
    """SRW02 (WalkCount)."""
    return _safe(_srw, mol, 2)


def SRW03(mol):
    """SRW03 (WalkCount)."""
    return _safe(_srw, mol, 3)


def SRW04(mol):
    """SRW04 (WalkCount)."""
    return _safe(_srw, mol, 4)


def SRW05(mol):
    """SRW05 (WalkCount)."""
    return _safe(_srw, mol, 5)


def SRW06(mol):
    """SRW06 (WalkCount)."""
    return _safe(_srw, mol, 6)


def SRW07(mol):
    """SRW07 (WalkCount)."""
    return _safe(_srw, mol, 7)


def SRW08(mol):
    """SRW08 (WalkCount)."""
    return _safe(_srw, mol, 8)


def SRW09(mol):
    """SRW09 (WalkCount)."""
    return _safe(_srw, mol, 9)


def SRW10(mol):
    """SRW10 (WalkCount)."""
    return _safe(_srw, mol, 10)


def TSRW10(mol):
    """TSRW10 (WalkCount)."""
    return _safe(_tsrw10, mol)


def MPC2(mol):
    """MPC2 (PathCount)."""
    return _safe(_mpc, mol, 2)


def MPC3(mol):
    """MPC3 (PathCount)."""
    return _safe(_mpc, mol, 3)


def MPC4(mol):
    """MPC4 (PathCount)."""
    return _safe(_mpc, mol, 4)


def MPC5(mol):
    """MPC5 (PathCount)."""
    return _safe(_mpc, mol, 5)


def MPC6(mol):
    """MPC6 (PathCount)."""
    return _safe(_mpc, mol, 6)


def MPC7(mol):
    """MPC7 (PathCount)."""
    return _safe(_mpc, mol, 7)


def MPC8(mol):
    """MPC8 (PathCount)."""
    return _safe(_mpc, mol, 8)


def MPC9(mol):
    """MPC9 (PathCount)."""
    return _safe(_mpc, mol, 9)


def MPC10(mol):
    """MPC10 (PathCount)."""
    return _safe(_mpc, mol, 10)


def TMPC10(mol):
    """TMPC10 (PathCount)."""
    return _safe(_tmpc10, mol)


def piPC1(mol):
    """piPC1 (PathCount)."""
    return _safe(_pipc, mol, 1)


def piPC2(mol):
    """piPC2 (PathCount)."""
    return _safe(_pipc, mol, 2)


def piPC3(mol):
    """piPC3 (PathCount)."""
    return _safe(_pipc, mol, 3)


def piPC4(mol):
    """piPC4 (PathCount)."""
    return _safe(_pipc, mol, 4)


def piPC5(mol):
    """piPC5 (PathCount)."""
    return _safe(_pipc, mol, 5)


def piPC6(mol):
    """piPC6 (PathCount)."""
    return _safe(_pipc, mol, 6)


def piPC7(mol):
    """piPC7 (PathCount)."""
    return _safe(_pipc, mol, 7)


def piPC8(mol):
    """piPC8 (PathCount)."""
    return _safe(_pipc, mol, 8)


def piPC9(mol):
    """piPC9 (PathCount)."""
    return _safe(_pipc, mol, 9)


def piPC10(mol):
    """piPC10 (PathCount)."""
    return _safe(_pipc, mol, 10)


def TpiPC10(mol):
    """TpiPC10 (PathCount)."""
    return _safe(_tpipc10, mol)


def DetourIndex(mol):
    """DetourIndex (DetourIndex)."""
    return _safe(_detour_index, mol)


def VAdjMat(mol):
    """VAdjMat (VertexAdjacencyInformation)."""
    return _safe(_vadjmat, mol)


def IC0(mol):
    """IC0 (InformationContent)."""
    return _safe(_ic, mol, 0)


def TIC0(mol):
    """TIC0 (InformationContent)."""
    return _safe(_tic, mol, 0)


def SIC0(mol):
    """SIC0 (InformationContent)."""
    return _safe(_sic, mol, 0)


def BIC0(mol):
    """BIC0 (InformationContent)."""
    return _safe(_bic, mol, 0)


def CIC0(mol):
    """CIC0 (InformationContent)."""
    return _safe(_cic, mol, 0)


def MIC0(mol):
    """MIC0 (InformationContent)."""
    return _safe(_mic, mol, 0)


def ZMIC0(mol):
    """ZMIC0 (InformationContent)."""
    return _safe(_zmic, mol, 0)


def IC1(mol):
    """IC1 (InformationContent)."""
    return _safe(_ic, mol, 1)


def TIC1(mol):
    """TIC1 (InformationContent)."""
    return _safe(_tic, mol, 1)


def SIC1(mol):
    """SIC1 (InformationContent)."""
    return _safe(_sic, mol, 1)


def BIC1(mol):
    """BIC1 (InformationContent)."""
    return _safe(_bic, mol, 1)


def CIC1(mol):
    """CIC1 (InformationContent)."""
    return _safe(_cic, mol, 1)


def MIC1(mol):
    """MIC1 (InformationContent)."""
    return _safe(_mic, mol, 1)


def ZMIC1(mol):
    """ZMIC1 (InformationContent)."""
    return _safe(_zmic, mol, 1)


def IC2(mol):
    """IC2 (InformationContent)."""
    return _safe(_ic, mol, 2)


def TIC2(mol):
    """TIC2 (InformationContent)."""
    return _safe(_tic, mol, 2)


def SIC2(mol):
    """SIC2 (InformationContent)."""
    return _safe(_sic, mol, 2)


def BIC2(mol):
    """BIC2 (InformationContent)."""
    return _safe(_bic, mol, 2)


def CIC2(mol):
    """CIC2 (InformationContent)."""
    return _safe(_cic, mol, 2)


def MIC2(mol):
    """MIC2 (InformationContent)."""
    return _safe(_mic, mol, 2)


def ZMIC2(mol):
    """ZMIC2 (InformationContent)."""
    return _safe(_zmic, mol, 2)


def IC3(mol):
    """IC3 (InformationContent)."""
    return _safe(_ic, mol, 3)


def TIC3(mol):
    """TIC3 (InformationContent)."""
    return _safe(_tic, mol, 3)


def SIC3(mol):
    """SIC3 (InformationContent)."""
    return _safe(_sic, mol, 3)


def BIC3(mol):
    """BIC3 (InformationContent)."""
    return _safe(_bic, mol, 3)


def CIC3(mol):
    """CIC3 (InformationContent)."""
    return _safe(_cic, mol, 3)


def MIC3(mol):
    """MIC3 (InformationContent)."""
    return _safe(_mic, mol, 3)


def ZMIC3(mol):
    """ZMIC3 (InformationContent)."""
    return _safe(_zmic, mol, 3)


def IC4(mol):
    """IC4 (InformationContent)."""
    return _safe(_ic, mol, 4)


def TIC4(mol):
    """TIC4 (InformationContent)."""
    return _safe(_tic, mol, 4)


def SIC4(mol):
    """SIC4 (InformationContent)."""
    return _safe(_sic, mol, 4)


def BIC4(mol):
    """BIC4 (InformationContent)."""
    return _safe(_bic, mol, 4)


def CIC4(mol):
    """CIC4 (InformationContent)."""
    return _safe(_cic, mol, 4)


def MIC4(mol):
    """MIC4 (InformationContent)."""
    return _safe(_mic, mol, 4)


def ZMIC4(mol):
    """ZMIC4 (InformationContent)."""
    return _safe(_zmic, mol, 4)


def IC5(mol):
    """IC5 (InformationContent)."""
    return _safe(_ic, mol, 5)


def TIC5(mol):
    """TIC5 (InformationContent)."""
    return _safe(_tic, mol, 5)


def SIC5(mol):
    """SIC5 (InformationContent)."""
    return _safe(_sic, mol, 5)


def BIC5(mol):
    """BIC5 (InformationContent)."""
    return _safe(_bic, mol, 5)


def CIC5(mol):
    """CIC5 (InformationContent)."""
    return _safe(_cic, mol, 5)


def MIC5(mol):
    """MIC5 (InformationContent)."""
    return _safe(_mic, mol, 5)


def ZMIC5(mol):
    """ZMIC5 (InformationContent)."""
    return _safe(_zmic, mol, 5)


ALL_DESCRIPTOR_NAMES = (
    "MWC01",
    "MWC02",
    "MWC03",
    "MWC04",
    "MWC05",
    "MWC06",
    "MWC07",
    "MWC08",
    "MWC09",
    "MWC10",
    "TMWC10",
    "SRW02",
    "SRW03",
    "SRW04",
    "SRW05",
    "SRW06",
    "SRW07",
    "SRW08",
    "SRW09",
    "SRW10",
    "TSRW10",
    "MPC2",
    "MPC3",
    "MPC4",
    "MPC5",
    "MPC6",
    "MPC7",
    "MPC8",
    "MPC9",
    "MPC10",
    "TMPC10",
    "piPC1",
    "piPC2",
    "piPC3",
    "piPC4",
    "piPC5",
    "piPC6",
    "piPC7",
    "piPC8",
    "piPC9",
    "piPC10",
    "TpiPC10",
    "DetourIndex",
    "VAdjMat",
    "IC0",
    "TIC0",
    "SIC0",
    "BIC0",
    "CIC0",
    "MIC0",
    "ZMIC0",
    "IC1",
    "TIC1",
    "SIC1",
    "BIC1",
    "CIC1",
    "MIC1",
    "ZMIC1",
    "IC2",
    "TIC2",
    "SIC2",
    "BIC2",
    "CIC2",
    "MIC2",
    "ZMIC2",
    "IC3",
    "TIC3",
    "SIC3",
    "BIC3",
    "CIC3",
    "MIC3",
    "ZMIC3",
    "IC4",
    "TIC4",
    "SIC4",
    "BIC4",
    "CIC4",
    "MIC4",
    "ZMIC4",
    "IC5",
    "TIC5",
    "SIC5",
    "BIC5",
    "CIC5",
    "MIC5",
    "ZMIC5",
)
