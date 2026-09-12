"""
Topological / connectivity indices for MILIA (descriptor programme, Pace 3 / v1.6.0).

First-party PLUGIN-NATIVE descriptors implemented from primary literature and validated
against an oracle: ``mordredcommunity`` (BSD-3) for the 104 Mordred-defined descriptors
(reproduced bit-exact, rtol <= 1e-4), and published reference values for the 3 Schultz
indices (no in-toolkit oracle). Standard zero-core-modification plugin contract
(``function_name`` resolves to ``name`` via plugin.yaml; NaN on any failure, never raise).

Families (107)
--------------
constitutional/topological (block):
    WienerIndex          WPath, WPol                                   Wiener 1947
    ZagrebIndex          Zagreb1/2, mZagreb1/2                         Gutman & Trinajstic 1972
    ABCIndex             ABC, ABCGG                                    Estrada 1998; Graovac-Ghorbani 2016
    EccentricConnectivity ECIndex                                      Sharma, Goswami & Madan 1997
    TopologicalIndex     Diameter, Radius, TopoShapeIndex, PetitjeanIndex   Petitjean 1992
    Schultz              MTI, SchultzIndex, ModSchultzIndex            Schultz 1989 (native, published-value gated)
    TopologicalCharge    GGI1-10, JGI1-10, JGT10                       Galvez et al. 1994 (10.1021/ci00019a008)
    MolecularDistanceEdge MDEC/MDEO/MDEN                               Liu, Cai & Yao 1998
    Chi                  Kier-Hall path/cluster/path-cluster/chain     Kier & Hall 1976/1986

All heavy-atom graph (explicit_hydrogens off), 2D (requires_3d=false), deterministic, pure
(RDKit + stdlib + numpy). NaN on invalid input / undefined value; boolean-free (all scalar).
FractionCSP3-style de-dup applied: the 5 valence-path low orders (Xp-0dv..Xp-4dv) that equal
the existing HAVE RDKit Chi0v..Chi4v are NOT shipped.

Author: Asadollah (Shahram) Boshra
License: MIT
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)
_PT = Chem.GetPeriodicTable()
_NAN = float("nan")


# --------------------------------------------------------------------------- helpers
def _matrices(mol):
    D = Chem.GetDistanceMatrix(mol).astype(float)
    A = Chem.GetAdjacencyMatrix(mol).astype(float)
    deg = A.sum(axis=0)
    ecc = D.max(axis=0)
    return D, A, deg, ecc


def _deltas(mol):
    n = mol.GetNumAtoms()
    d = np.zeros(n)
    dv = np.zeros(n)
    for a in mol.GetAtoms():
        i = a.GetIdx()
        d[i] = sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum() != 1)
        z = a.GetAtomicNum()
        zv = _PT.GetNOuterElecs(z) - a.GetFormalCharge()
        zc = z - a.GetFormalCharge()
        h = a.GetTotalNumHs()
        denom = zc - zv - 1
        dv[i] = (zv - h) / denom if denom != 0 else _NAN
    return d, dv


def _classify(nbrs):
    import sys

    sys.setrecursionlimit(100000)
    visited = set()
    vis_edges = set()
    is_chain = [False]
    degrees = set()

    def dfs(u):
        visited.add(u)
        degrees.add(len(nbrs[u]))
        for v in nbrs[u]:
            ek = (v, u) if u > v else (u, v)
            if v not in visited:
                vis_edges.add(ek)
                dfs(v)
            elif ek not in vis_edges:
                vis_edges.add(ek)
                is_chain[0] = True

    dfs(next(iter(nbrs)))
    if is_chain[0]:
        return "chain"
    if not (degrees - {1, 2}):
        return "path"
    if 2 in degrees:
        return "path_cluster"
    return "cluster"


def _chi_subgraphs(mol, order):
    res = {"path": [], "cluster": [], "path_cluster": [], "chain": []}
    if order == 0:
        res["path"] = [[a.GetIdx()] for a in mol.GetAtoms()]
        return res
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    for bidx in Chem.FindAllSubgraphsOfLengthN(mol, order):
        nbrs = defaultdict(set)
        for bi in bidx:
            a, b = bonds[bi]
            nbrs[a].add(b)
            nbrs[b].add(a)
        res[_classify(nbrs)].append(list(nbrs.keys()))
    return res


def _charge_terms(mol):
    D = Chem.GetDistanceMatrix(mol).astype(float)
    A = Chem.GetAdjacencyMatrix(mol).astype(float)
    d2 = D.copy()
    nz = d2 != 0
    d2[nz] = d2[nz] ** -2.0
    np.fill_diagonal(d2, 0)
    m = A.dot(d2)
    return m - m.T, D


# --------------------------------------------------------------------------- family impls
def _wiener(mol, polarity):
    D, _, _, _ = _matrices(mol)
    return float(int(0.5 * ((D == 3).sum() if polarity else D.sum())))


def _zagreb(mol, version, variable):
    _, _, d, _ = _matrices(mol)
    if version == 1:
        if variable < 0 and (d == 0).any():
            return _NAN
        return float((d ** (variable * 2)).sum())
    tot = 0.0
    for b in mol.GetBonds():
        p = d[b.GetBeginAtomIdx()] * d[b.GetEndAtomIdx()]
        if variable < 0:
            if p == 0:
                return _NAN
            tot += p**variable
        else:
            tot += p**variable
    return float(tot)


def _abc(mol):
    tot = 0.0
    for b in mol.GetBonds():
        du = b.GetBeginAtom().GetDegree()
        dv = b.GetEndAtom().GetDegree()
        tot += math.sqrt((du + dv - 2.0) / (du * dv))
    return float(tot)


def _abcgg(mol):
    D, _, _, _ = _matrices(mol)
    tot = 0.0
    for b in mol.GetBonds():
        u = b.GetBeginAtomIdx()
        v = b.GetEndAtomIdx()
        nu = int((D[u, :] < D[v, :]).sum())
        nv = int((D[v, :] < D[u, :]).sum())
        tot += math.sqrt((nu + nv - 2.0) / (nu * nv))
    return float(tot)


def _ecindex(mol):
    D, A, d, ecc = _matrices(mol)
    return float(int((ecc.astype(int) * d).sum()))


def _radius(mol):
    _, _, _, ecc = _matrices(mol)
    return float(int(ecc.min()))


def _diameter(mol):
    _, _, _, ecc = _matrices(mol)
    return float(int(ecc.max()))


def _topo_shape(mol):
    _, _, _, ecc = _matrices(mol)
    r = ecc.min()
    return float((ecc.max() - r) / r) if r else _NAN


def _petitjean(mol):
    _, _, _, ecc = _matrices(mol)
    dia = ecc.max()
    return float((dia - ecc.min()) / dia) if dia else _NAN


def _mti(mol):
    D, A, d, _ = _matrices(mol)
    return float((d @ (A + D)).sum())


def _schultz(mol):
    D, _, d, _ = _matrices(mol)
    return float(0.25 * ((d[:, None] + d[None, :]) * D).sum())


def _mod_schultz(mol):
    D, _, d, _ = _matrices(mol)
    return float(0.25 * ((d[:, None] * d[None, :]) * D).sum())


def _chi(mol, typ, order, prop, averaged):
    d, dv = _deltas(mol)
    P = d if prop == "d" else dv
    sg = _chi_subgraphs(mol, order)[typ]
    x = 0.0
    for nodes in sg:
        c = 1.0
        for nd in nodes:
            c *= P[nd]
        if not (c > 0):
            return _NAN
        x += c**-0.5
    if averaged:
        if len(sg) == 0:
            return _NAN
        x /= len(sg)
    return float(x)


def _topo_charge(mol, typ, order):
    ct, D = _charge_terms(mol)
    dl = D * np.tri(*D.shape)
    dl[dl == 0] = np.inf
    f = (dl <= order) if typ == "global" else (dl == order)
    ctf = ct[f]
    if typ == "raw":
        return float(np.abs(ctf).sum())
    df = dl[f]
    c = df.copy()
    for i in np.unique(df):
        c[df == i] = len(df[df == i])
    if len(ctf) == 0:
        return 0.0
    return float(np.abs(ctf / c).sum())


def _mde(mol, valence1, valence2, atomic_num):
    D = Chem.GetDistanceMatrix(mol).astype(float)
    V = Chem.GetAdjacencyMatrix(mol).sum(axis=0)
    n_at = mol.GetNumAtoms()
    v1, v2 = min(valence1, valence2), max(valence1, valence2)
    dv = []
    for i in range(n_at):
        for j in range(i + 1, n_at):
            if ((V[i] == v1 and V[j] == v2) or (V[j] == v1 and V[i] == v2)) and (
                mol.GetAtomWithIdx(i).GetAtomicNum()
                == atomic_num
                == mol.GetAtomWithIdx(j).GetAtomicNum()
            ):
                dv.append(D[i, j])
    n = len(dv)
    if n == 0:
        return _NAN
    return float(n / math.exp(sum(math.log(x) for x in dv) / n))


def _safe(fn, *args):
    try:
        v = fn(*args)
        v = float(v)
        return v if math.isfinite(v) or math.isnan(v) else _NAN
    except Exception:
        return _NAN


# --------------------------------------------------------------------------- descriptors
def WPath(mol):
    """WPath (WienerIndex)."""
    return _safe(_wiener, mol, False)


def WPol(mol):
    """WPol (WienerIndex)."""
    return _safe(_wiener, mol, True)


def Zagreb1(mol):
    """Zagreb1 (ZagrebIndex)."""
    return _safe(_zagreb, mol, 1, 1)


def Zagreb2(mol):
    """Zagreb2 (ZagrebIndex)."""
    return _safe(_zagreb, mol, 2, 1)


def mZagreb1(mol):
    """mZagreb1 (ZagrebIndex)."""
    return _safe(_zagreb, mol, 1, -1)


def mZagreb2(mol):
    """mZagreb2 (ZagrebIndex)."""
    return _safe(_zagreb, mol, 2, -1)


def ABC(mol):
    """ABC (ABCIndex)."""
    return _safe(_abc, mol)


def ABCGG(mol):
    """ABCGG (ABCIndex)."""
    return _safe(_abcgg, mol)


def ECIndex(mol):
    """ECIndex (EccentricConnectivity)."""
    return _safe(_ecindex, mol)


def Diameter(mol):
    """Diameter (TopologicalIndex)."""
    return _safe(_diameter, mol)


def Radius(mol):
    """Radius (TopologicalIndex)."""
    return _safe(_radius, mol)


def TopoShapeIndex(mol):
    """TopoShapeIndex (TopologicalIndex)."""
    return _safe(_topo_shape, mol)


def PetitjeanIndex(mol):
    """PetitjeanIndex (TopologicalIndex)."""
    return _safe(_petitjean, mol)


def MTI(mol):
    """MTI (Schultz)."""
    return _safe(_mti, mol)


def SchultzIndex(mol):
    """SchultzIndex (Schultz)."""
    return _safe(_schultz, mol)


def ModSchultzIndex(mol):
    """ModSchultzIndex (Schultz)."""
    return _safe(_mod_schultz, mol)


def Xch_3d(mol):
    """Xch-3d (Chi)."""
    return _safe(_chi, mol, "chain", 3, "d", False)


def Xch_4d(mol):
    """Xch-4d (Chi)."""
    return _safe(_chi, mol, "chain", 4, "d", False)


def Xch_5d(mol):
    """Xch-5d (Chi)."""
    return _safe(_chi, mol, "chain", 5, "d", False)


def Xch_6d(mol):
    """Xch-6d (Chi)."""
    return _safe(_chi, mol, "chain", 6, "d", False)


def Xch_7d(mol):
    """Xch-7d (Chi)."""
    return _safe(_chi, mol, "chain", 7, "d", False)


def Xc_3d(mol):
    """Xc-3d (Chi)."""
    return _safe(_chi, mol, "cluster", 3, "d", False)


def Xc_4d(mol):
    """Xc-4d (Chi)."""
    return _safe(_chi, mol, "cluster", 4, "d", False)


def Xc_5d(mol):
    """Xc-5d (Chi)."""
    return _safe(_chi, mol, "cluster", 5, "d", False)


def Xc_6d(mol):
    """Xc-6d (Chi)."""
    return _safe(_chi, mol, "cluster", 6, "d", False)


def Xpc_4d(mol):
    """Xpc-4d (Chi)."""
    return _safe(_chi, mol, "path_cluster", 4, "d", False)


def Xpc_5d(mol):
    """Xpc-5d (Chi)."""
    return _safe(_chi, mol, "path_cluster", 5, "d", False)


def Xpc_6d(mol):
    """Xpc-6d (Chi)."""
    return _safe(_chi, mol, "path_cluster", 6, "d", False)


def Xp_0d(mol):
    """Xp-0d (Chi)."""
    return _safe(_chi, mol, "path", 0, "d", False)


def AXp_0d(mol):
    """AXp-0d (Chi)."""
    return _safe(_chi, mol, "path", 0, "d", True)


def Xp_1d(mol):
    """Xp-1d (Chi)."""
    return _safe(_chi, mol, "path", 1, "d", False)


def AXp_1d(mol):
    """AXp-1d (Chi)."""
    return _safe(_chi, mol, "path", 1, "d", True)


def Xp_2d(mol):
    """Xp-2d (Chi)."""
    return _safe(_chi, mol, "path", 2, "d", False)


def AXp_2d(mol):
    """AXp-2d (Chi)."""
    return _safe(_chi, mol, "path", 2, "d", True)


def Xp_3d(mol):
    """Xp-3d (Chi)."""
    return _safe(_chi, mol, "path", 3, "d", False)


def AXp_3d(mol):
    """AXp-3d (Chi)."""
    return _safe(_chi, mol, "path", 3, "d", True)


def Xp_4d(mol):
    """Xp-4d (Chi)."""
    return _safe(_chi, mol, "path", 4, "d", False)


def AXp_4d(mol):
    """AXp-4d (Chi)."""
    return _safe(_chi, mol, "path", 4, "d", True)


def Xp_5d(mol):
    """Xp-5d (Chi)."""
    return _safe(_chi, mol, "path", 5, "d", False)


def AXp_5d(mol):
    """AXp-5d (Chi)."""
    return _safe(_chi, mol, "path", 5, "d", True)


def Xp_6d(mol):
    """Xp-6d (Chi)."""
    return _safe(_chi, mol, "path", 6, "d", False)


def AXp_6d(mol):
    """AXp-6d (Chi)."""
    return _safe(_chi, mol, "path", 6, "d", True)


def Xp_7d(mol):
    """Xp-7d (Chi)."""
    return _safe(_chi, mol, "path", 7, "d", False)


def AXp_7d(mol):
    """AXp-7d (Chi)."""
    return _safe(_chi, mol, "path", 7, "d", True)


def Xch_3dv(mol):
    """Xch-3dv (Chi)."""
    return _safe(_chi, mol, "chain", 3, "dv", False)


def Xch_4dv(mol):
    """Xch-4dv (Chi)."""
    return _safe(_chi, mol, "chain", 4, "dv", False)


def Xch_5dv(mol):
    """Xch-5dv (Chi)."""
    return _safe(_chi, mol, "chain", 5, "dv", False)


def Xch_6dv(mol):
    """Xch-6dv (Chi)."""
    return _safe(_chi, mol, "chain", 6, "dv", False)


def Xch_7dv(mol):
    """Xch-7dv (Chi)."""
    return _safe(_chi, mol, "chain", 7, "dv", False)


def Xc_3dv(mol):
    """Xc-3dv (Chi)."""
    return _safe(_chi, mol, "cluster", 3, "dv", False)


def Xc_4dv(mol):
    """Xc-4dv (Chi)."""
    return _safe(_chi, mol, "cluster", 4, "dv", False)


def Xc_5dv(mol):
    """Xc-5dv (Chi)."""
    return _safe(_chi, mol, "cluster", 5, "dv", False)


def Xc_6dv(mol):
    """Xc-6dv (Chi)."""
    return _safe(_chi, mol, "cluster", 6, "dv", False)


def Xpc_4dv(mol):
    """Xpc-4dv (Chi)."""
    return _safe(_chi, mol, "path_cluster", 4, "dv", False)


def Xpc_5dv(mol):
    """Xpc-5dv (Chi)."""
    return _safe(_chi, mol, "path_cluster", 5, "dv", False)


def Xpc_6dv(mol):
    """Xpc-6dv (Chi)."""
    return _safe(_chi, mol, "path_cluster", 6, "dv", False)


def AXp_0dv(mol):
    """AXp-0dv (Chi)."""
    return _safe(_chi, mol, "path", 0, "dv", True)


def AXp_1dv(mol):
    """AXp-1dv (Chi)."""
    return _safe(_chi, mol, "path", 1, "dv", True)


def AXp_2dv(mol):
    """AXp-2dv (Chi)."""
    return _safe(_chi, mol, "path", 2, "dv", True)


def AXp_3dv(mol):
    """AXp-3dv (Chi)."""
    return _safe(_chi, mol, "path", 3, "dv", True)


def AXp_4dv(mol):
    """AXp-4dv (Chi)."""
    return _safe(_chi, mol, "path", 4, "dv", True)


def Xp_5dv(mol):
    """Xp-5dv (Chi)."""
    return _safe(_chi, mol, "path", 5, "dv", False)


def AXp_5dv(mol):
    """AXp-5dv (Chi)."""
    return _safe(_chi, mol, "path", 5, "dv", True)


def Xp_6dv(mol):
    """Xp-6dv (Chi)."""
    return _safe(_chi, mol, "path", 6, "dv", False)


def AXp_6dv(mol):
    """AXp-6dv (Chi)."""
    return _safe(_chi, mol, "path", 6, "dv", True)


def Xp_7dv(mol):
    """Xp-7dv (Chi)."""
    return _safe(_chi, mol, "path", 7, "dv", False)


def AXp_7dv(mol):
    """AXp-7dv (Chi)."""
    return _safe(_chi, mol, "path", 7, "dv", True)


def GGI1(mol):
    """GGI1 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "raw", 1)


def GGI2(mol):
    """GGI2 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "raw", 2)


def GGI3(mol):
    """GGI3 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "raw", 3)


def GGI4(mol):
    """GGI4 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "raw", 4)


def GGI5(mol):
    """GGI5 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "raw", 5)


def GGI6(mol):
    """GGI6 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "raw", 6)


def GGI7(mol):
    """GGI7 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "raw", 7)


def GGI8(mol):
    """GGI8 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "raw", 8)


def GGI9(mol):
    """GGI9 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "raw", 9)


def GGI10(mol):
    """GGI10 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "raw", 10)


def JGI1(mol):
    """JGI1 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "mean", 1)


def JGI2(mol):
    """JGI2 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "mean", 2)


def JGI3(mol):
    """JGI3 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "mean", 3)


def JGI4(mol):
    """JGI4 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "mean", 4)


def JGI5(mol):
    """JGI5 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "mean", 5)


def JGI6(mol):
    """JGI6 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "mean", 6)


def JGI7(mol):
    """JGI7 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "mean", 7)


def JGI8(mol):
    """JGI8 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "mean", 8)


def JGI9(mol):
    """JGI9 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "mean", 9)


def JGI10(mol):
    """JGI10 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "mean", 10)


def JGT10(mol):
    """JGT10 (TopologicalCharge)."""
    return _safe(_topo_charge, mol, "global", 10)


def MDEC_11(mol):
    """MDEC-11 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 1, 1, 6)


def MDEC_12(mol):
    """MDEC-12 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 1, 2, 6)


def MDEC_13(mol):
    """MDEC-13 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 1, 3, 6)


def MDEC_14(mol):
    """MDEC-14 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 1, 4, 6)


def MDEC_22(mol):
    """MDEC-22 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 2, 2, 6)


def MDEC_23(mol):
    """MDEC-23 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 2, 3, 6)


def MDEC_24(mol):
    """MDEC-24 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 2, 4, 6)


def MDEC_33(mol):
    """MDEC-33 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 3, 3, 6)


def MDEC_34(mol):
    """MDEC-34 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 3, 4, 6)


def MDEC_44(mol):
    """MDEC-44 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 4, 4, 6)


def MDEO_11(mol):
    """MDEO-11 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 1, 1, 8)


def MDEO_12(mol):
    """MDEO-12 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 1, 2, 8)


def MDEO_22(mol):
    """MDEO-22 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 2, 2, 8)


def MDEN_11(mol):
    """MDEN-11 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 1, 1, 7)


def MDEN_12(mol):
    """MDEN-12 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 1, 2, 7)


def MDEN_13(mol):
    """MDEN-13 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 1, 3, 7)


def MDEN_22(mol):
    """MDEN-22 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 2, 2, 7)


def MDEN_23(mol):
    """MDEN-23 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 2, 3, 7)


def MDEN_33(mol):
    """MDEN-33 (MolecularDistanceEdge)."""
    return _safe(_mde, mol, 3, 3, 7)


ALL_DESCRIPTOR_NAMES = (
    "WPath",
    "WPol",
    "Zagreb1",
    "Zagreb2",
    "mZagreb1",
    "mZagreb2",
    "ABC",
    "ABCGG",
    "ECIndex",
    "Diameter",
    "Radius",
    "TopoShapeIndex",
    "PetitjeanIndex",
    "MTI",
    "SchultzIndex",
    "ModSchultzIndex",
    "Xch_3d",
    "Xch_4d",
    "Xch_5d",
    "Xch_6d",
    "Xch_7d",
    "Xc_3d",
    "Xc_4d",
    "Xc_5d",
    "Xc_6d",
    "Xpc_4d",
    "Xpc_5d",
    "Xpc_6d",
    "Xp_0d",
    "AXp_0d",
    "Xp_1d",
    "AXp_1d",
    "Xp_2d",
    "AXp_2d",
    "Xp_3d",
    "AXp_3d",
    "Xp_4d",
    "AXp_4d",
    "Xp_5d",
    "AXp_5d",
    "Xp_6d",
    "AXp_6d",
    "Xp_7d",
    "AXp_7d",
    "Xch_3dv",
    "Xch_4dv",
    "Xch_5dv",
    "Xch_6dv",
    "Xch_7dv",
    "Xc_3dv",
    "Xc_4dv",
    "Xc_5dv",
    "Xc_6dv",
    "Xpc_4dv",
    "Xpc_5dv",
    "Xpc_6dv",
    "AXp_0dv",
    "AXp_1dv",
    "AXp_2dv",
    "AXp_3dv",
    "AXp_4dv",
    "Xp_5dv",
    "AXp_5dv",
    "Xp_6dv",
    "AXp_6dv",
    "Xp_7dv",
    "AXp_7dv",
    "GGI1",
    "GGI2",
    "GGI3",
    "GGI4",
    "GGI5",
    "GGI6",
    "GGI7",
    "GGI8",
    "GGI9",
    "GGI10",
    "JGI1",
    "JGI2",
    "JGI3",
    "JGI4",
    "JGI5",
    "JGI6",
    "JGI7",
    "JGI8",
    "JGI9",
    "JGI10",
    "JGT10",
    "MDEC_11",
    "MDEC_12",
    "MDEC_13",
    "MDEC_14",
    "MDEC_22",
    "MDEC_23",
    "MDEC_24",
    "MDEC_33",
    "MDEC_34",
    "MDEC_44",
    "MDEO_11",
    "MDEO_12",
    "MDEO_22",
    "MDEN_11",
    "MDEN_12",
    "MDEN_13",
    "MDEN_22",
    "MDEN_23",
    "MDEN_33",
)
