"""
Charged partial surface area (CPSA), gravitational-index, and geometrical-index 3D
descriptors for MILIA (descriptor programme, Pace 9 / v1.12.0).

First-party PLUGIN-NATIVE 3D descriptors computed on the MILIA-core conformer
(AddHs -> ETKDG EmbedMolecule(randomSeed=42) -> MMFFOptimizeMolecule). Validated bit-exact
against the ``mordredcommunity`` oracle (rel-err 0; 50/50 on the panel). Zero-core-modification
plugin contract (``name`` == ``function_name``; NaN on any failure, never raise).

Families (50):
  * CPSA (42) - Stanton & Jurs, Anal. Chem. 1990, 62, 2323 (doi:10.1021/ac00220a013):
    partial/fractional/surface-weighted negative & positive surface areas (PNSA/PPSA/DPSA/
    FNSA/FPSA/WNSA/WPSA, versions 1-5), relative charges (RNCG/RPCG), relative charged
    surface areas (RNCS/RPCS), and hydrophobic/polar surface areas (TASA/RASA/RPSA).
    Surface area = Shrake-Rupley dot tessellation over a level-5 icosphere (vdW + 1.4 A
    solvent radius); charges = RDKit Gasteiger.
  * GravitationalIndex (4) - GRAV/GRAVH/GRAVp/GRAVHp (mass-weighted 1/r^2 sums).
  * GeometricalIndex (4) - geometric diameter/radius/shape/Petitjean (3D distance matrix).

De-dup: the mordred CPSA ``TPSA`` (3D total polar surface area) is NOT shipped - its name
collides with the HAVE RDKit topological ``TPSA`` (registry name-uniqueness); it is still
computed internally as it feeds ``RPSA``. ``MomentOfInertia`` (= HAVE ``PMI1/2/3``) and
``PBF`` (= HAVE ``PBF``) are excluded as duplicates.

Author: Asadollah (Shahram) Boshra
License: MIT
"""

from __future__ import annotations

import functools
import logging
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

logger = logging.getLogger(__name__)
_NAN = float("nan")

# van der Waals radii (Angstrom), Z -> radius (mordredcommunity table)
_VDW = {
    1: 1.1,
    2: 1.4,
    3: 1.82,
    4: 1.53,
    5: 1.92,
    6: 1.7,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    10: 1.54,
    11: 2.27,
    12: 1.73,
    13: 1.84,
    14: 2.1,
    15: 1.8,
    16: 1.8,
    17: 1.75,
    18: 1.88,
    19: 2.75,
    20: 2.31,
    21: 2.15,
    22: 2.11,
    23: 2.07,
    24: 2.06,
    25: 2.05,
    26: 2.04,
    27: 2.0,
    28: 1.97,
    29: 1.96,
    30: 2.01,
    31: 1.87,
    32: 2.11,
    33: 1.85,
    34: 1.9,
    35: 1.85,
    36: 2.02,
    37: 3.03,
    38: 2.49,
    39: 2.32,
    40: 2.23,
    41: 2.18,
    42: 2.17,
    43: 2.16,
    44: 2.13,
    45: 2.1,
    46: 2.1,
    47: 2.11,
    48: 2.18,
    49: 1.93,
    50: 2.17,
    51: 2.06,
    52: 2.06,
    53: 1.98,
    54: 2.16,
    55: 3.43,
    56: 2.68,
    57: 2.43,
    58: 2.42,
    59: 2.4,
    60: 2.39,
    61: 2.38,
    62: 2.36,
    63: 2.35,
    64: 2.34,
    65: 2.33,
    66: 2.31,
    67: 2.3,
    68: 2.29,
    69: 2.27,
    70: 2.26,
    71: 2.24,
    72: 2.23,
    73: 2.22,
    74: 2.18,
    75: 2.16,
    76: 2.16,
    77: 2.13,
    78: 2.13,
    79: 2.14,
    80: 2.23,
    81: 1.96,
    82: 2.02,
    83: 2.07,
    84: 1.97,
    85: 2.02,
    86: 2.2,
    87: 3.48,
    88: 2.83,
    89: 2.47,
    90: 2.45,
    91: 2.43,
    92: 2.41,
    93: 2.39,
    94: 2.43,
    95: 2.44,
    96: 2.45,
    97: 2.44,
    98: 2.45,
    99: 2.45,
    100: 2.45,
    101: 2.46,
    102: 2.46,
    103: 2.46,
}


class _SphereMesh:
    """Non-deduplicated icosphere (mordred.surface_area._mesh, verbatim)."""

    def __init__(self, level=4):
        t = (1.0 + np.sqrt(5.0)) / 2.0
        self.vertices = np.array(
            [
                (-1, t, 0),
                (1, t, 0),
                (-1, -t, 0),
                (1, -t, 0),
                (0, -1, t),
                (0, 1, t),
                (0, -1, -t),
                (0, 1, -t),
                (t, 0, -1),
                (t, 0, 1),
                (-t, 0, -1),
                (-t, 0, 1),
            ],
            dtype="float",
        )
        self.faces = np.array(
            [
                (0, 11, 5),
                (0, 5, 1),
                (0, 1, 7),
                (0, 7, 10),
                (0, 10, 11),
                (1, 5, 9),
                (5, 11, 4),
                (11, 10, 2),
                (10, 7, 6),
                (7, 1, 8),
                (3, 9, 4),
                (3, 4, 2),
                (3, 2, 6),
                (3, 6, 8),
                (3, 8, 9),
                (4, 9, 5),
                (2, 4, 11),
                (6, 2, 10),
                (8, 6, 7),
                (9, 8, 1),
            ],
            dtype="int",
        )
        self.normalize(0)
        self.level = 1
        self.subdivide(level)

    def normalize(self, begin):
        self.vertices[begin:] /= np.sqrt((self.vertices[begin:] ** 2).sum(axis=1))[:, np.newaxis]

    def _subdivide(self):
        self.level += 1
        nv, nf = len(self.vertices), len(self.faces)
        a, b, c = self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]
        av, bv, cv = self.vertices[a], self.vertices[b], self.vertices[c]
        self.vertices = np.r_[self.vertices, av + bv, bv + cv, av + cv]
        self.normalize(nv)
        ab = np.arange(len(self.faces)) + nv
        bc, ac = ab + nf, ab + 2 * nf
        self.faces = np.concatenate([a, b, c, ab, ab, ab, ac, ac, ac, bc, bc, bc]).reshape(3, -1).T

    def subdivide(self, level):
        for _ in range(level - self.level):
            self._subdivide()


@functools.lru_cache(maxsize=1)
def _mesh_level5():
    return _SphereMesh(5).vertices.T


class _SurfaceArea:
    """Shrake-Rupley SASA (mordred.surface_area._sasa, verbatim)."""

    def __init__(self, radiuses, xyzs, level=5):
        self.rads = radiuses
        self.rads2 = radiuses**2
        self.xyzs = xyzs
        self._gen_neighbor_list()
        self.sphere = _mesh_level5()

    def _gen_neighbor_list(self):
        r = self.rads[:, np.newaxis] + self.rads
        d = np.sqrt(np.sum((self.xyzs[:, np.newaxis] - self.xyzs) ** 2, axis=2))
        ns = defaultdict(list)
        for i, j in np.transpose(np.nonzero(d <= r)):
            if i == j:
                continue
            ns[i].append((j, d[i, j]))
        for _, lst in ns.items():
            lst.sort(key=lambda x: x[1])
        self.neighbors = ns

    def atomic_sa(self, i):
        sa = 4.0 * np.pi * self.rads2[i]
        neighbors = self.neighbors.get(i)
        if neighbors is None:
            return sa
        xyzi = self.xyzs[i, np.newaxis].T
        sphere = self.sphere * self.rads[i] + xyzi
        n = sphere.shape[1]
        for j, _ in neighbors:
            xyzj = self.xyzs[j, np.newaxis].T
            d2 = (sphere - xyzj) ** 2
            mask = (d2[0] + d2[1] + d2[2]) > self.rads2[j]
            sphere = np.compress(mask, sphere, axis=1)
        return sa * sphere.shape[1] / n

    def surface_area(self):
        return np.array([self.atomic_sa(i) for i in range(len(self.rads))])


def _coords(mol):
    conf = mol.GetConformer(-1)
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])


def _atomic_sa(mol, solvent_radius=1.4, level=5):
    rs = np.array([_VDW[a.GetAtomicNum()] + solvent_radius for a in mol.GetAtoms()])
    return _SurfaceArea(rs, _coords(mol), level).surface_area()


def _gasteiger(mol):
    rdPartialCharges.ComputeGasteigerCharges(mol)
    out = []
    for a in mol.GetAtoms():
        v = a.GetDoubleProp("_GasteigerCharge")
        if a.HasProp("_GasteigerHCharge"):
            v += a.GetDoubleProp("_GasteigerHCharge")
        out.append(v)
    return np.array(out)


@functools.lru_cache(maxsize=1)
def _block(mol):
    """Compute all 50 CPSA/Grav/Geom descriptors once per molecule (single-slot memoised).

    Requires a conformer (embedded by the MILIA core). Any failure -> all NaN.
    """
    out = {name: _NAN for name in ALL_DESCRIPTOR_NAMES}
    try:
        molh = Chem.Mol(mol)
        molh.GetConformer(-1)
    except Exception:
        return out

    # ---- CPSA ----
    try:
        sa = _atomic_sa(molh)
        q = _gasteiger(molh)
        tot = float(np.sum(sa))
        nat = molh.GetNumAtoms()

        def _pnsa(ver, pos):
            mask = q > 0.0 if pos else q < 0.0
            if ver == 1:
                f = 1.0
            elif ver == 2:
                f = np.sum(q[mask])
            elif ver == 3:
                f = q[mask]
            elif ver == 4:
                f = np.sum(q[mask]) / nat
            else:
                f = np.sum(q[mask]) / np.sum(mask)
            return float(np.sum(f * sa[mask]))

        cpsa = {}
        for v in range(1, 6):
            cpsa[f"PNSA{v}"] = _pnsa(v, False)
            cpsa[f"PPSA{v}"] = _pnsa(v, True)
            cpsa[f"DPSA{v}"] = cpsa[f"PPSA{v}"] - cpsa[f"PNSA{v}"]
            cpsa[f"FNSA{v}"] = cpsa[f"PNSA{v}"] / tot
            cpsa[f"FPSA{v}"] = cpsa[f"PPSA{v}"] / tot
            cpsa[f"WNSA{v}"] = cpsa[f"PNSA{v}"] * tot / 1000.0
            cpsa[f"WPSA{v}"] = cpsa[f"PPSA{v}"] * tot / 1000.0

        def _rcg(pos):
            m = q > 0.0 if pos else q < 0.0
            c = q[m]
            if len(c) == 0:
                return 0.0
            return float(c[np.argmax(np.abs(c))] / np.sum(c))

        cpsa["RNCG"] = _rcg(False)
        cpsa["RPCG"] = _rcg(True)

        def _rcs(pos):
            m = q > 0.0 if pos else q < 0.0
            c = q[m]
            if len(c) == 0:
                return 0.0
            return float(sa[m][np.argmax(np.abs(c))] / _rcg(pos))

        cpsa["RNCS"] = _rcs(False)
        cpsa["RPCS"] = _rcs(True)
        cpsa["TASA"] = float(np.sum(sa[np.abs(q) < 0.2]))
        _tpsa = float(np.sum(sa[np.abs(q) >= 0.2]))  # internal only (name-clash) -> feeds RPSA
        cpsa["RASA"] = cpsa["TASA"] / tot
        cpsa["RPSA"] = _tpsa / tot
        for k, v in cpsa.items():
            if k in out:
                out[k] = v
    except Exception:
        pass

    # ---- GravitationalIndex ----
    try:
        molheavy = Chem.RemoveHs(molh)
        for heavy in (True, False):
            m = molheavy if heavy else molh
            w = np.array([a.GetMass() for a in m.GetAtoms()])
            w = w[:, np.newaxis] * w
            np.fill_diagonal(w, 0)
            crd = _coords(m)
            d = np.sqrt(np.sum((crd[:, np.newaxis] - crd) ** 2, axis=2)).copy()
            np.fill_diagonal(d, 1)
            adj = Chem.GetAdjacencyMatrix(m).astype(float)
            for pair in (False, True):
                a_mat = adj if pair else 1.0
                name = "GRAV" + ("" if heavy else "H") + ("p" if pair else "")
                out[name] = float(0.5 * np.sum(w * a_mat / d**2))
    except Exception:
        pass

    # ---- GeometricalIndex ----
    try:
        crd = _coords(molh)
        d = np.sqrt(np.sum((crd[:, np.newaxis] - crd) ** 2, axis=2))
        ecc = d.max(axis=0)
        r = float(ecc.min())
        dia = float(ecc.max())
        out["GeomDiameter"] = dia
        out["GeomRadius"] = r
        out["GeomShapeIndex"] = (dia - r) / r if r != 0 else _NAN
        out["GeomPetitjeanIndex"] = (dia - r) / dia if dia != 0 else _NAN
    except Exception:
        pass

    return out


def _make(name):
    def fn(mol):
        try:
            if mol is None:
                return _NAN
            return float(_block(mol)[name])
        except Exception:
            return _NAN

    return fn


_PNSA1 = _make("PNSA1")
_PNSA2 = _make("PNSA2")
_PNSA3 = _make("PNSA3")
_PNSA4 = _make("PNSA4")
_PNSA5 = _make("PNSA5")
_PPSA1 = _make("PPSA1")
_PPSA2 = _make("PPSA2")
_PPSA3 = _make("PPSA3")
_PPSA4 = _make("PPSA4")
_PPSA5 = _make("PPSA5")
_DPSA1 = _make("DPSA1")
_DPSA2 = _make("DPSA2")
_DPSA3 = _make("DPSA3")
_DPSA4 = _make("DPSA4")
_DPSA5 = _make("DPSA5")
_FNSA1 = _make("FNSA1")
_FNSA2 = _make("FNSA2")
_FNSA3 = _make("FNSA3")
_FNSA4 = _make("FNSA4")
_FNSA5 = _make("FNSA5")
_FPSA1 = _make("FPSA1")
_FPSA2 = _make("FPSA2")
_FPSA3 = _make("FPSA3")
_FPSA4 = _make("FPSA4")
_FPSA5 = _make("FPSA5")
_WNSA1 = _make("WNSA1")
_WNSA2 = _make("WNSA2")
_WNSA3 = _make("WNSA3")
_WNSA4 = _make("WNSA4")
_WNSA5 = _make("WNSA5")
_WPSA1 = _make("WPSA1")
_WPSA2 = _make("WPSA2")
_WPSA3 = _make("WPSA3")
_WPSA4 = _make("WPSA4")
_WPSA5 = _make("WPSA5")
_RNCG = _make("RNCG")
_RPCG = _make("RPCG")
_RNCS = _make("RNCS")
_RPCS = _make("RPCS")
_TASA = _make("TASA")
_RASA = _make("RASA")
_RPSA = _make("RPSA")
_GRAV = _make("GRAV")
_GRAVH = _make("GRAVH")
_GRAVp = _make("GRAVp")
_GRAVHp = _make("GRAVHp")
_GeomDiameter = _make("GeomDiameter")
_GeomRadius = _make("GeomRadius")
_GeomShapeIndex = _make("GeomShapeIndex")
_GeomPetitjeanIndex = _make("GeomPetitjeanIndex")


# ---- public wrappers (name == function_name; declared -> 0 bonus) ----


def PNSA1(mol):
    """partial negative surface area (version 1)."""
    return _PNSA1(mol)


def PNSA2(mol):
    """partial negative surface area (version 2)."""
    return _PNSA2(mol)


def PNSA3(mol):
    """partial negative surface area (version 3)."""
    return _PNSA3(mol)


def PNSA4(mol):
    """partial negative surface area (version 4)."""
    return _PNSA4(mol)


def PNSA5(mol):
    """partial negative surface area (version 5)."""
    return _PNSA5(mol)


def PPSA1(mol):
    """partial positive surface area (version 1)."""
    return _PPSA1(mol)


def PPSA2(mol):
    """partial positive surface area (version 2)."""
    return _PPSA2(mol)


def PPSA3(mol):
    """partial positive surface area (version 3)."""
    return _PPSA3(mol)


def PPSA4(mol):
    """partial positive surface area (version 4)."""
    return _PPSA4(mol)


def PPSA5(mol):
    """partial positive surface area (version 5)."""
    return _PPSA5(mol)


def DPSA1(mol):
    """difference in charged partial surface area (version 1)."""
    return _DPSA1(mol)


def DPSA2(mol):
    """difference in charged partial surface area (version 2)."""
    return _DPSA2(mol)


def DPSA3(mol):
    """difference in charged partial surface area (version 3)."""
    return _DPSA3(mol)


def DPSA4(mol):
    """difference in charged partial surface area (version 4)."""
    return _DPSA4(mol)


def DPSA5(mol):
    """difference in charged partial surface area (version 5)."""
    return _DPSA5(mol)


def FNSA1(mol):
    """fractional charged partial negative surface area (version 1)."""
    return _FNSA1(mol)


def FNSA2(mol):
    """fractional charged partial negative surface area (version 2)."""
    return _FNSA2(mol)


def FNSA3(mol):
    """fractional charged partial negative surface area (version 3)."""
    return _FNSA3(mol)


def FNSA4(mol):
    """fractional charged partial negative surface area (version 4)."""
    return _FNSA4(mol)


def FNSA5(mol):
    """fractional charged partial negative surface area (version 5)."""
    return _FNSA5(mol)


def FPSA1(mol):
    """fractional charged partial positive surface area (version 1)."""
    return _FPSA1(mol)


def FPSA2(mol):
    """fractional charged partial positive surface area (version 2)."""
    return _FPSA2(mol)


def FPSA3(mol):
    """fractional charged partial positive surface area (version 3)."""
    return _FPSA3(mol)


def FPSA4(mol):
    """fractional charged partial positive surface area (version 4)."""
    return _FPSA4(mol)


def FPSA5(mol):
    """fractional charged partial positive surface area (version 5)."""
    return _FPSA5(mol)


def WNSA1(mol):
    """surface-weighted charged partial negative surface area (version 1)."""
    return _WNSA1(mol)


def WNSA2(mol):
    """surface-weighted charged partial negative surface area (version 2)."""
    return _WNSA2(mol)


def WNSA3(mol):
    """surface-weighted charged partial negative surface area (version 3)."""
    return _WNSA3(mol)


def WNSA4(mol):
    """surface-weighted charged partial negative surface area (version 4)."""
    return _WNSA4(mol)


def WNSA5(mol):
    """surface-weighted charged partial negative surface area (version 5)."""
    return _WNSA5(mol)


def WPSA1(mol):
    """surface-weighted charged partial positive surface area (version 1)."""
    return _WPSA1(mol)


def WPSA2(mol):
    """surface-weighted charged partial positive surface area (version 2)."""
    return _WPSA2(mol)


def WPSA3(mol):
    """surface-weighted charged partial positive surface area (version 3)."""
    return _WPSA3(mol)


def WPSA4(mol):
    """surface-weighted charged partial positive surface area (version 4)."""
    return _WPSA4(mol)


def WPSA5(mol):
    """surface-weighted charged partial positive surface area (version 5)."""
    return _WPSA5(mol)


def RNCG(mol):
    """relative negative charge."""
    return _RNCG(mol)


def RPCG(mol):
    """relative positive charge."""
    return _RPCG(mol)


def RNCS(mol):
    """relative negative charge surface area."""
    return _RNCS(mol)


def RPCS(mol):
    """relative positive charge surface area."""
    return _RPCS(mol)


def TASA(mol):
    """total hydrophobic surface area."""
    return _TASA(mol)


def RASA(mol):
    """relative hydrophobic surface area."""
    return _RASA(mol)


def RPSA(mol):
    """relative polar surface area."""
    return _RPSA(mol)


def GRAV(mol):
    """gravitational index."""
    return _GRAV(mol)


def GRAVH(mol):
    """gravitational index (incl. H)."""
    return _GRAVH(mol)


def GRAVp(mol):
    """gravitational index (bonded pairs)."""
    return _GRAVp(mol)


def GRAVHp(mol):
    """gravitational index (incl. H, bonded pairs)."""
    return _GRAVHp(mol)


def GeomDiameter(mol):
    """geometric diameter."""
    return _GeomDiameter(mol)


def GeomRadius(mol):
    """geometric radius."""
    return _GeomRadius(mol)


def GeomShapeIndex(mol):
    """geometrical shape index."""
    return _GeomShapeIndex(mol)


def GeomPetitjeanIndex(mol):
    """geometric Petitjean index."""
    return _GeomPetitjeanIndex(mol)


ALL_DESCRIPTOR_NAMES = (
    "PNSA1",
    "PNSA2",
    "PNSA3",
    "PNSA4",
    "PNSA5",
    "PPSA1",
    "PPSA2",
    "PPSA3",
    "PPSA4",
    "PPSA5",
    "DPSA1",
    "DPSA2",
    "DPSA3",
    "DPSA4",
    "DPSA5",
    "FNSA1",
    "FNSA2",
    "FNSA3",
    "FNSA4",
    "FNSA5",
    "FPSA1",
    "FPSA2",
    "FPSA3",
    "FPSA4",
    "FPSA5",
    "WNSA1",
    "WNSA2",
    "WNSA3",
    "WNSA4",
    "WNSA5",
    "WPSA1",
    "WPSA2",
    "WPSA3",
    "WPSA4",
    "WPSA5",
    "RNCG",
    "RPCG",
    "RNCS",
    "RPCS",
    "TASA",
    "RASA",
    "RPSA",
    "GRAV",
    "GRAVH",
    "GRAVp",
    "GRAVHp",
    "GeomDiameter",
    "GeomRadius",
    "GeomShapeIndex",
    "GeomPetitjeanIndex",
)
