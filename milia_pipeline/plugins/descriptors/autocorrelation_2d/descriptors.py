"""
2D topological autocorrelation descriptors for MILIA (descriptor programme, Pace 6 / v1.9.0).

First-party PLUGIN-NATIVE descriptors implemented from primary literature and validated
bit-exact against the ``mordredcommunity`` oracle (rel-err 0.0). Zero-core-modification
plugin contract (``name`` == ``function_name``; NaN on any failure, never raise).

Families (606) — all on the H-included heavy-atom graph (explicit_hydrogens=True):
    ATS   (99)   Moreau-Broto autocorrelation (w^T B_k w)           Moreau & Broto 1980
    AATS  (99)   Averaged Moreau-Broto (ATS_k / Delta_k)            Moreau & Broto 1980
    ATSC  (108)  Centred Moreau-Broto (centred property vector)      Todeschini 2009
    AATSC (108)  Averaged + centred Moreau-Broto                     Todeschini 2009
    MATS  (96)   Moran coefficient (N * AATSC_k / var(w_c))          Moran 1950
    GATS  (96)   Geary coefficient                                    Geary 1954

Property weightings:
    ATS/AATS (11 props × lags 0-8): Z m v se pe are p i d dv
    ATSC/AATSC/MATS/GATS (12 props × lags 0-8 or 1-8): + c (Gasteiger charge)
    Static tables (Z=atomic number; m,v,se,pe,are,p,i = primary-literature values embedded here).
    Dynamic (molecule-computed): c (Gasteiger), d (heavy sigma-degree), dv (Kier-Hall delta-v),
    s (intrinsic state — note: not in ATS/AATS presets).

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

# --- static atomic-property tables (primary-literature values, Pace-5 provenance) ---
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
_STATIC = {
    "m": _MASS,
    "v": _VDWV,
    "se": _SANDERSON,
    "pe": _PAULING,
    "are": _ALLRED_ROCHOW,
    "p": _POLARIZ,
    "i": _IONIZ,
}


# --------------------------------------------------------------------------- property vectors
def _sigma_deg(a):
    return sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum() != 1)


def _val_delta(a):
    N = a.GetAtomicNum()
    if N == 1:
        return 0.0
    Zv = _PT.GetNOuterElecs(N) - a.GetFormalCharge()
    Z = N - a.GetFormalCharge()
    h = a.GetTotalNumHs() + sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum() == 1)
    denom = Z - Zv - 1
    return (Zv - h) / denom if denom else _NAN


def _intr_state(a):
    d = _sigma_deg(a)
    if d == 0:
        return _NAN
    per = _PERIOD.get(a.GetAtomicNum())
    return ((2.0 / per) ** 2 * _val_delta(a) + 1) / d if per else _NAN


def _gasteiger(mol):
    mm = Chem.Mol(mol)
    rdPartialCharges.ComputeGasteigerCharges(mm)
    out = []
    for a in mm.GetAtoms():
        v = a.GetDoubleProp("_GasteigerCharge")
        if a.HasProp("_GasteigerHCharge"):
            v += a.GetDoubleProp("_GasteigerHCharge")
        out.append(v)
    return np.array(out)


def _prop_vec(mol, short):
    mh = Chem.AddHs(Chem.Mol(mol))
    if short == "Z":
        return np.array([float(a.GetAtomicNum()) for a in mh.GetAtoms()])
    if short == "d":
        return np.array([float(_sigma_deg(a)) for a in mh.GetAtoms()])
    if short == "dv":
        return np.array([float(_val_delta(a)) for a in mh.GetAtoms()])
    if short == "s":
        return np.array([float(_intr_state(a)) for a in mh.GetAtoms()])
    if short == "c":
        return _gasteiger(mh)
    tbl = _STATIC[short]
    return np.array([float(tbl.get(a.GetAtomicNum(), _NAN)) for a in mh.GetAtoms()])


# --------------------------------------------------------------------------- autocorrelation engine
def _autocorr(mol, kind, lag, short):
    mh = Chem.AddHs(Chem.Mol(mol))
    w = _prop_vec(mol, short)
    if np.any(np.isnan(w)):
        return _NAN
    n = len(w)
    D = Chem.GetDistanceMatrix(mh).astype(float)
    centred = kind in ("ATSC", "AATSC", "MATS", "GATS")
    wc = w - w.mean() if centred else w
    if kind in ("ATS", "AATS", "ATSC", "AATSC"):
        if lag == 0:
            v0 = float((wc**2).sum())
            if kind in ("AATS", "AATSC"):
                return v0 / n if n else _NAN
            return v0
        G = (lag == D).astype(float)
        gs = 0.5 * G.sum()
        v = float(0.5 * wc.dot(G).dot(wc))
        if kind in ("AATS", "AATSC"):
            return v / gs if gs else _NAN
        return v
    # MATS / GATS (lag >= 1)
    if lag == 0:
        return _NAN
    var_wc = float((wc**2).sum())
    if var_wc == 0:
        return _NAN
    G = (lag == D).astype(float)
    gs = float(0.5 * G.sum())
    if gs == 0:
        return _NAN
    if kind == "MATS":
        atsc_k = float(0.5 * wc.dot(G).dot(wc))
        return float(n * atsc_k / gs / var_wc)
    # GATS
    if n <= 1:
        return _NAN
    num = float((G * (w[:, None] - w) ** 2).sum()) / (4 * gs)
    den = var_wc / (n - 1)
    return num / den if den else _NAN


def _safe_ac(kind, lag, short):
    def fn(mol):
        try:
            if mol is None:
                return _NAN
            v = _autocorr(mol, kind, lag, short)
            return v if math.isfinite(v) or math.isnan(v) else _NAN
        except Exception:
            return _NAN

    return fn


# --------------------------------------------------------------------------- descriptors
_ATS0dv = _safe_ac("ATS", 0, "dv")
_ATS1dv = _safe_ac("ATS", 1, "dv")
_ATS2dv = _safe_ac("ATS", 2, "dv")
_ATS3dv = _safe_ac("ATS", 3, "dv")
_ATS4dv = _safe_ac("ATS", 4, "dv")
_ATS5dv = _safe_ac("ATS", 5, "dv")
_ATS6dv = _safe_ac("ATS", 6, "dv")
_ATS7dv = _safe_ac("ATS", 7, "dv")
_ATS8dv = _safe_ac("ATS", 8, "dv")
_ATS0d = _safe_ac("ATS", 0, "d")
_ATS1d = _safe_ac("ATS", 1, "d")
_ATS2d = _safe_ac("ATS", 2, "d")
_ATS3d = _safe_ac("ATS", 3, "d")
_ATS4d = _safe_ac("ATS", 4, "d")
_ATS5d = _safe_ac("ATS", 5, "d")
_ATS6d = _safe_ac("ATS", 6, "d")
_ATS7d = _safe_ac("ATS", 7, "d")
_ATS8d = _safe_ac("ATS", 8, "d")
_ATS0s = _safe_ac("ATS", 0, "s")
_ATS1s = _safe_ac("ATS", 1, "s")
_ATS2s = _safe_ac("ATS", 2, "s")
_ATS3s = _safe_ac("ATS", 3, "s")
_ATS4s = _safe_ac("ATS", 4, "s")
_ATS5s = _safe_ac("ATS", 5, "s")
_ATS6s = _safe_ac("ATS", 6, "s")
_ATS7s = _safe_ac("ATS", 7, "s")
_ATS8s = _safe_ac("ATS", 8, "s")
_ATS0Z = _safe_ac("ATS", 0, "Z")
_ATS1Z = _safe_ac("ATS", 1, "Z")
_ATS2Z = _safe_ac("ATS", 2, "Z")
_ATS3Z = _safe_ac("ATS", 3, "Z")
_ATS4Z = _safe_ac("ATS", 4, "Z")
_ATS5Z = _safe_ac("ATS", 5, "Z")
_ATS6Z = _safe_ac("ATS", 6, "Z")
_ATS7Z = _safe_ac("ATS", 7, "Z")
_ATS8Z = _safe_ac("ATS", 8, "Z")
_ATS0m = _safe_ac("ATS", 0, "m")
_ATS1m = _safe_ac("ATS", 1, "m")
_ATS2m = _safe_ac("ATS", 2, "m")
_ATS3m = _safe_ac("ATS", 3, "m")
_ATS4m = _safe_ac("ATS", 4, "m")
_ATS5m = _safe_ac("ATS", 5, "m")
_ATS6m = _safe_ac("ATS", 6, "m")
_ATS7m = _safe_ac("ATS", 7, "m")
_ATS8m = _safe_ac("ATS", 8, "m")
_ATS0v = _safe_ac("ATS", 0, "v")
_ATS1v = _safe_ac("ATS", 1, "v")
_ATS2v = _safe_ac("ATS", 2, "v")
_ATS3v = _safe_ac("ATS", 3, "v")
_ATS4v = _safe_ac("ATS", 4, "v")
_ATS5v = _safe_ac("ATS", 5, "v")
_ATS6v = _safe_ac("ATS", 6, "v")
_ATS7v = _safe_ac("ATS", 7, "v")
_ATS8v = _safe_ac("ATS", 8, "v")
_ATS0se = _safe_ac("ATS", 0, "se")
_ATS1se = _safe_ac("ATS", 1, "se")
_ATS2se = _safe_ac("ATS", 2, "se")
_ATS3se = _safe_ac("ATS", 3, "se")
_ATS4se = _safe_ac("ATS", 4, "se")
_ATS5se = _safe_ac("ATS", 5, "se")
_ATS6se = _safe_ac("ATS", 6, "se")
_ATS7se = _safe_ac("ATS", 7, "se")
_ATS8se = _safe_ac("ATS", 8, "se")
_ATS0pe = _safe_ac("ATS", 0, "pe")
_ATS1pe = _safe_ac("ATS", 1, "pe")
_ATS2pe = _safe_ac("ATS", 2, "pe")
_ATS3pe = _safe_ac("ATS", 3, "pe")
_ATS4pe = _safe_ac("ATS", 4, "pe")
_ATS5pe = _safe_ac("ATS", 5, "pe")
_ATS6pe = _safe_ac("ATS", 6, "pe")
_ATS7pe = _safe_ac("ATS", 7, "pe")
_ATS8pe = _safe_ac("ATS", 8, "pe")
_ATS0are = _safe_ac("ATS", 0, "are")
_ATS1are = _safe_ac("ATS", 1, "are")
_ATS2are = _safe_ac("ATS", 2, "are")
_ATS3are = _safe_ac("ATS", 3, "are")
_ATS4are = _safe_ac("ATS", 4, "are")
_ATS5are = _safe_ac("ATS", 5, "are")
_ATS6are = _safe_ac("ATS", 6, "are")
_ATS7are = _safe_ac("ATS", 7, "are")
_ATS8are = _safe_ac("ATS", 8, "are")
_ATS0p = _safe_ac("ATS", 0, "p")
_ATS1p = _safe_ac("ATS", 1, "p")
_ATS2p = _safe_ac("ATS", 2, "p")
_ATS3p = _safe_ac("ATS", 3, "p")
_ATS4p = _safe_ac("ATS", 4, "p")
_ATS5p = _safe_ac("ATS", 5, "p")
_ATS6p = _safe_ac("ATS", 6, "p")
_ATS7p = _safe_ac("ATS", 7, "p")
_ATS8p = _safe_ac("ATS", 8, "p")
_ATS0i = _safe_ac("ATS", 0, "i")
_ATS1i = _safe_ac("ATS", 1, "i")
_ATS2i = _safe_ac("ATS", 2, "i")
_ATS3i = _safe_ac("ATS", 3, "i")
_ATS4i = _safe_ac("ATS", 4, "i")
_ATS5i = _safe_ac("ATS", 5, "i")
_ATS6i = _safe_ac("ATS", 6, "i")
_ATS7i = _safe_ac("ATS", 7, "i")
_ATS8i = _safe_ac("ATS", 8, "i")
_AATS0dv = _safe_ac("AATS", 0, "dv")
_AATS1dv = _safe_ac("AATS", 1, "dv")
_AATS2dv = _safe_ac("AATS", 2, "dv")
_AATS3dv = _safe_ac("AATS", 3, "dv")
_AATS4dv = _safe_ac("AATS", 4, "dv")
_AATS5dv = _safe_ac("AATS", 5, "dv")
_AATS6dv = _safe_ac("AATS", 6, "dv")
_AATS7dv = _safe_ac("AATS", 7, "dv")
_AATS8dv = _safe_ac("AATS", 8, "dv")
_AATS0d = _safe_ac("AATS", 0, "d")
_AATS1d = _safe_ac("AATS", 1, "d")
_AATS2d = _safe_ac("AATS", 2, "d")
_AATS3d = _safe_ac("AATS", 3, "d")
_AATS4d = _safe_ac("AATS", 4, "d")
_AATS5d = _safe_ac("AATS", 5, "d")
_AATS6d = _safe_ac("AATS", 6, "d")
_AATS7d = _safe_ac("AATS", 7, "d")
_AATS8d = _safe_ac("AATS", 8, "d")
_AATS0s = _safe_ac("AATS", 0, "s")
_AATS1s = _safe_ac("AATS", 1, "s")
_AATS2s = _safe_ac("AATS", 2, "s")
_AATS3s = _safe_ac("AATS", 3, "s")
_AATS4s = _safe_ac("AATS", 4, "s")
_AATS5s = _safe_ac("AATS", 5, "s")
_AATS6s = _safe_ac("AATS", 6, "s")
_AATS7s = _safe_ac("AATS", 7, "s")
_AATS8s = _safe_ac("AATS", 8, "s")
_AATS0Z = _safe_ac("AATS", 0, "Z")
_AATS1Z = _safe_ac("AATS", 1, "Z")
_AATS2Z = _safe_ac("AATS", 2, "Z")
_AATS3Z = _safe_ac("AATS", 3, "Z")
_AATS4Z = _safe_ac("AATS", 4, "Z")
_AATS5Z = _safe_ac("AATS", 5, "Z")
_AATS6Z = _safe_ac("AATS", 6, "Z")
_AATS7Z = _safe_ac("AATS", 7, "Z")
_AATS8Z = _safe_ac("AATS", 8, "Z")
_AATS0m = _safe_ac("AATS", 0, "m")
_AATS1m = _safe_ac("AATS", 1, "m")
_AATS2m = _safe_ac("AATS", 2, "m")
_AATS3m = _safe_ac("AATS", 3, "m")
_AATS4m = _safe_ac("AATS", 4, "m")
_AATS5m = _safe_ac("AATS", 5, "m")
_AATS6m = _safe_ac("AATS", 6, "m")
_AATS7m = _safe_ac("AATS", 7, "m")
_AATS8m = _safe_ac("AATS", 8, "m")
_AATS0v = _safe_ac("AATS", 0, "v")
_AATS1v = _safe_ac("AATS", 1, "v")
_AATS2v = _safe_ac("AATS", 2, "v")
_AATS3v = _safe_ac("AATS", 3, "v")
_AATS4v = _safe_ac("AATS", 4, "v")
_AATS5v = _safe_ac("AATS", 5, "v")
_AATS6v = _safe_ac("AATS", 6, "v")
_AATS7v = _safe_ac("AATS", 7, "v")
_AATS8v = _safe_ac("AATS", 8, "v")
_AATS0se = _safe_ac("AATS", 0, "se")
_AATS1se = _safe_ac("AATS", 1, "se")
_AATS2se = _safe_ac("AATS", 2, "se")
_AATS3se = _safe_ac("AATS", 3, "se")
_AATS4se = _safe_ac("AATS", 4, "se")
_AATS5se = _safe_ac("AATS", 5, "se")
_AATS6se = _safe_ac("AATS", 6, "se")
_AATS7se = _safe_ac("AATS", 7, "se")
_AATS8se = _safe_ac("AATS", 8, "se")
_AATS0pe = _safe_ac("AATS", 0, "pe")
_AATS1pe = _safe_ac("AATS", 1, "pe")
_AATS2pe = _safe_ac("AATS", 2, "pe")
_AATS3pe = _safe_ac("AATS", 3, "pe")
_AATS4pe = _safe_ac("AATS", 4, "pe")
_AATS5pe = _safe_ac("AATS", 5, "pe")
_AATS6pe = _safe_ac("AATS", 6, "pe")
_AATS7pe = _safe_ac("AATS", 7, "pe")
_AATS8pe = _safe_ac("AATS", 8, "pe")
_AATS0are = _safe_ac("AATS", 0, "are")
_AATS1are = _safe_ac("AATS", 1, "are")
_AATS2are = _safe_ac("AATS", 2, "are")
_AATS3are = _safe_ac("AATS", 3, "are")
_AATS4are = _safe_ac("AATS", 4, "are")
_AATS5are = _safe_ac("AATS", 5, "are")
_AATS6are = _safe_ac("AATS", 6, "are")
_AATS7are = _safe_ac("AATS", 7, "are")
_AATS8are = _safe_ac("AATS", 8, "are")
_AATS0p = _safe_ac("AATS", 0, "p")
_AATS1p = _safe_ac("AATS", 1, "p")
_AATS2p = _safe_ac("AATS", 2, "p")
_AATS3p = _safe_ac("AATS", 3, "p")
_AATS4p = _safe_ac("AATS", 4, "p")
_AATS5p = _safe_ac("AATS", 5, "p")
_AATS6p = _safe_ac("AATS", 6, "p")
_AATS7p = _safe_ac("AATS", 7, "p")
_AATS8p = _safe_ac("AATS", 8, "p")
_AATS0i = _safe_ac("AATS", 0, "i")
_AATS1i = _safe_ac("AATS", 1, "i")
_AATS2i = _safe_ac("AATS", 2, "i")
_AATS3i = _safe_ac("AATS", 3, "i")
_AATS4i = _safe_ac("AATS", 4, "i")
_AATS5i = _safe_ac("AATS", 5, "i")
_AATS6i = _safe_ac("AATS", 6, "i")
_AATS7i = _safe_ac("AATS", 7, "i")
_AATS8i = _safe_ac("AATS", 8, "i")
_ATSC0c = _safe_ac("ATSC", 0, "c")
_ATSC1c = _safe_ac("ATSC", 1, "c")
_ATSC2c = _safe_ac("ATSC", 2, "c")
_ATSC3c = _safe_ac("ATSC", 3, "c")
_ATSC4c = _safe_ac("ATSC", 4, "c")
_ATSC5c = _safe_ac("ATSC", 5, "c")
_ATSC6c = _safe_ac("ATSC", 6, "c")
_ATSC7c = _safe_ac("ATSC", 7, "c")
_ATSC8c = _safe_ac("ATSC", 8, "c")
_ATSC0dv = _safe_ac("ATSC", 0, "dv")
_ATSC1dv = _safe_ac("ATSC", 1, "dv")
_ATSC2dv = _safe_ac("ATSC", 2, "dv")
_ATSC3dv = _safe_ac("ATSC", 3, "dv")
_ATSC4dv = _safe_ac("ATSC", 4, "dv")
_ATSC5dv = _safe_ac("ATSC", 5, "dv")
_ATSC6dv = _safe_ac("ATSC", 6, "dv")
_ATSC7dv = _safe_ac("ATSC", 7, "dv")
_ATSC8dv = _safe_ac("ATSC", 8, "dv")
_ATSC0d = _safe_ac("ATSC", 0, "d")
_ATSC1d = _safe_ac("ATSC", 1, "d")
_ATSC2d = _safe_ac("ATSC", 2, "d")
_ATSC3d = _safe_ac("ATSC", 3, "d")
_ATSC4d = _safe_ac("ATSC", 4, "d")
_ATSC5d = _safe_ac("ATSC", 5, "d")
_ATSC6d = _safe_ac("ATSC", 6, "d")
_ATSC7d = _safe_ac("ATSC", 7, "d")
_ATSC8d = _safe_ac("ATSC", 8, "d")
_ATSC0s = _safe_ac("ATSC", 0, "s")
_ATSC1s = _safe_ac("ATSC", 1, "s")
_ATSC2s = _safe_ac("ATSC", 2, "s")
_ATSC3s = _safe_ac("ATSC", 3, "s")
_ATSC4s = _safe_ac("ATSC", 4, "s")
_ATSC5s = _safe_ac("ATSC", 5, "s")
_ATSC6s = _safe_ac("ATSC", 6, "s")
_ATSC7s = _safe_ac("ATSC", 7, "s")
_ATSC8s = _safe_ac("ATSC", 8, "s")
_ATSC0Z = _safe_ac("ATSC", 0, "Z")
_ATSC1Z = _safe_ac("ATSC", 1, "Z")
_ATSC2Z = _safe_ac("ATSC", 2, "Z")
_ATSC3Z = _safe_ac("ATSC", 3, "Z")
_ATSC4Z = _safe_ac("ATSC", 4, "Z")
_ATSC5Z = _safe_ac("ATSC", 5, "Z")
_ATSC6Z = _safe_ac("ATSC", 6, "Z")
_ATSC7Z = _safe_ac("ATSC", 7, "Z")
_ATSC8Z = _safe_ac("ATSC", 8, "Z")
_ATSC0m = _safe_ac("ATSC", 0, "m")
_ATSC1m = _safe_ac("ATSC", 1, "m")
_ATSC2m = _safe_ac("ATSC", 2, "m")
_ATSC3m = _safe_ac("ATSC", 3, "m")
_ATSC4m = _safe_ac("ATSC", 4, "m")
_ATSC5m = _safe_ac("ATSC", 5, "m")
_ATSC6m = _safe_ac("ATSC", 6, "m")
_ATSC7m = _safe_ac("ATSC", 7, "m")
_ATSC8m = _safe_ac("ATSC", 8, "m")
_ATSC0v = _safe_ac("ATSC", 0, "v")
_ATSC1v = _safe_ac("ATSC", 1, "v")
_ATSC2v = _safe_ac("ATSC", 2, "v")
_ATSC3v = _safe_ac("ATSC", 3, "v")
_ATSC4v = _safe_ac("ATSC", 4, "v")
_ATSC5v = _safe_ac("ATSC", 5, "v")
_ATSC6v = _safe_ac("ATSC", 6, "v")
_ATSC7v = _safe_ac("ATSC", 7, "v")
_ATSC8v = _safe_ac("ATSC", 8, "v")
_ATSC0se = _safe_ac("ATSC", 0, "se")
_ATSC1se = _safe_ac("ATSC", 1, "se")
_ATSC2se = _safe_ac("ATSC", 2, "se")
_ATSC3se = _safe_ac("ATSC", 3, "se")
_ATSC4se = _safe_ac("ATSC", 4, "se")
_ATSC5se = _safe_ac("ATSC", 5, "se")
_ATSC6se = _safe_ac("ATSC", 6, "se")
_ATSC7se = _safe_ac("ATSC", 7, "se")
_ATSC8se = _safe_ac("ATSC", 8, "se")
_ATSC0pe = _safe_ac("ATSC", 0, "pe")
_ATSC1pe = _safe_ac("ATSC", 1, "pe")
_ATSC2pe = _safe_ac("ATSC", 2, "pe")
_ATSC3pe = _safe_ac("ATSC", 3, "pe")
_ATSC4pe = _safe_ac("ATSC", 4, "pe")
_ATSC5pe = _safe_ac("ATSC", 5, "pe")
_ATSC6pe = _safe_ac("ATSC", 6, "pe")
_ATSC7pe = _safe_ac("ATSC", 7, "pe")
_ATSC8pe = _safe_ac("ATSC", 8, "pe")
_ATSC0are = _safe_ac("ATSC", 0, "are")
_ATSC1are = _safe_ac("ATSC", 1, "are")
_ATSC2are = _safe_ac("ATSC", 2, "are")
_ATSC3are = _safe_ac("ATSC", 3, "are")
_ATSC4are = _safe_ac("ATSC", 4, "are")
_ATSC5are = _safe_ac("ATSC", 5, "are")
_ATSC6are = _safe_ac("ATSC", 6, "are")
_ATSC7are = _safe_ac("ATSC", 7, "are")
_ATSC8are = _safe_ac("ATSC", 8, "are")
_ATSC0p = _safe_ac("ATSC", 0, "p")
_ATSC1p = _safe_ac("ATSC", 1, "p")
_ATSC2p = _safe_ac("ATSC", 2, "p")
_ATSC3p = _safe_ac("ATSC", 3, "p")
_ATSC4p = _safe_ac("ATSC", 4, "p")
_ATSC5p = _safe_ac("ATSC", 5, "p")
_ATSC6p = _safe_ac("ATSC", 6, "p")
_ATSC7p = _safe_ac("ATSC", 7, "p")
_ATSC8p = _safe_ac("ATSC", 8, "p")
_ATSC0i = _safe_ac("ATSC", 0, "i")
_ATSC1i = _safe_ac("ATSC", 1, "i")
_ATSC2i = _safe_ac("ATSC", 2, "i")
_ATSC3i = _safe_ac("ATSC", 3, "i")
_ATSC4i = _safe_ac("ATSC", 4, "i")
_ATSC5i = _safe_ac("ATSC", 5, "i")
_ATSC6i = _safe_ac("ATSC", 6, "i")
_ATSC7i = _safe_ac("ATSC", 7, "i")
_ATSC8i = _safe_ac("ATSC", 8, "i")
_AATSC0c = _safe_ac("AATSC", 0, "c")
_AATSC1c = _safe_ac("AATSC", 1, "c")
_AATSC2c = _safe_ac("AATSC", 2, "c")
_AATSC3c = _safe_ac("AATSC", 3, "c")
_AATSC4c = _safe_ac("AATSC", 4, "c")
_AATSC5c = _safe_ac("AATSC", 5, "c")
_AATSC6c = _safe_ac("AATSC", 6, "c")
_AATSC7c = _safe_ac("AATSC", 7, "c")
_AATSC8c = _safe_ac("AATSC", 8, "c")
_AATSC0dv = _safe_ac("AATSC", 0, "dv")
_AATSC1dv = _safe_ac("AATSC", 1, "dv")
_AATSC2dv = _safe_ac("AATSC", 2, "dv")
_AATSC3dv = _safe_ac("AATSC", 3, "dv")
_AATSC4dv = _safe_ac("AATSC", 4, "dv")
_AATSC5dv = _safe_ac("AATSC", 5, "dv")
_AATSC6dv = _safe_ac("AATSC", 6, "dv")
_AATSC7dv = _safe_ac("AATSC", 7, "dv")
_AATSC8dv = _safe_ac("AATSC", 8, "dv")
_AATSC0d = _safe_ac("AATSC", 0, "d")
_AATSC1d = _safe_ac("AATSC", 1, "d")
_AATSC2d = _safe_ac("AATSC", 2, "d")
_AATSC3d = _safe_ac("AATSC", 3, "d")
_AATSC4d = _safe_ac("AATSC", 4, "d")
_AATSC5d = _safe_ac("AATSC", 5, "d")
_AATSC6d = _safe_ac("AATSC", 6, "d")
_AATSC7d = _safe_ac("AATSC", 7, "d")
_AATSC8d = _safe_ac("AATSC", 8, "d")
_AATSC0s = _safe_ac("AATSC", 0, "s")
_AATSC1s = _safe_ac("AATSC", 1, "s")
_AATSC2s = _safe_ac("AATSC", 2, "s")
_AATSC3s = _safe_ac("AATSC", 3, "s")
_AATSC4s = _safe_ac("AATSC", 4, "s")
_AATSC5s = _safe_ac("AATSC", 5, "s")
_AATSC6s = _safe_ac("AATSC", 6, "s")
_AATSC7s = _safe_ac("AATSC", 7, "s")
_AATSC8s = _safe_ac("AATSC", 8, "s")
_AATSC0Z = _safe_ac("AATSC", 0, "Z")
_AATSC1Z = _safe_ac("AATSC", 1, "Z")
_AATSC2Z = _safe_ac("AATSC", 2, "Z")
_AATSC3Z = _safe_ac("AATSC", 3, "Z")
_AATSC4Z = _safe_ac("AATSC", 4, "Z")
_AATSC5Z = _safe_ac("AATSC", 5, "Z")
_AATSC6Z = _safe_ac("AATSC", 6, "Z")
_AATSC7Z = _safe_ac("AATSC", 7, "Z")
_AATSC8Z = _safe_ac("AATSC", 8, "Z")
_AATSC0m = _safe_ac("AATSC", 0, "m")
_AATSC1m = _safe_ac("AATSC", 1, "m")
_AATSC2m = _safe_ac("AATSC", 2, "m")
_AATSC3m = _safe_ac("AATSC", 3, "m")
_AATSC4m = _safe_ac("AATSC", 4, "m")
_AATSC5m = _safe_ac("AATSC", 5, "m")
_AATSC6m = _safe_ac("AATSC", 6, "m")
_AATSC7m = _safe_ac("AATSC", 7, "m")
_AATSC8m = _safe_ac("AATSC", 8, "m")
_AATSC0v = _safe_ac("AATSC", 0, "v")
_AATSC1v = _safe_ac("AATSC", 1, "v")
_AATSC2v = _safe_ac("AATSC", 2, "v")
_AATSC3v = _safe_ac("AATSC", 3, "v")
_AATSC4v = _safe_ac("AATSC", 4, "v")
_AATSC5v = _safe_ac("AATSC", 5, "v")
_AATSC6v = _safe_ac("AATSC", 6, "v")
_AATSC7v = _safe_ac("AATSC", 7, "v")
_AATSC8v = _safe_ac("AATSC", 8, "v")
_AATSC0se = _safe_ac("AATSC", 0, "se")
_AATSC1se = _safe_ac("AATSC", 1, "se")
_AATSC2se = _safe_ac("AATSC", 2, "se")
_AATSC3se = _safe_ac("AATSC", 3, "se")
_AATSC4se = _safe_ac("AATSC", 4, "se")
_AATSC5se = _safe_ac("AATSC", 5, "se")
_AATSC6se = _safe_ac("AATSC", 6, "se")
_AATSC7se = _safe_ac("AATSC", 7, "se")
_AATSC8se = _safe_ac("AATSC", 8, "se")
_AATSC0pe = _safe_ac("AATSC", 0, "pe")
_AATSC1pe = _safe_ac("AATSC", 1, "pe")
_AATSC2pe = _safe_ac("AATSC", 2, "pe")
_AATSC3pe = _safe_ac("AATSC", 3, "pe")
_AATSC4pe = _safe_ac("AATSC", 4, "pe")
_AATSC5pe = _safe_ac("AATSC", 5, "pe")
_AATSC6pe = _safe_ac("AATSC", 6, "pe")
_AATSC7pe = _safe_ac("AATSC", 7, "pe")
_AATSC8pe = _safe_ac("AATSC", 8, "pe")
_AATSC0are = _safe_ac("AATSC", 0, "are")
_AATSC1are = _safe_ac("AATSC", 1, "are")
_AATSC2are = _safe_ac("AATSC", 2, "are")
_AATSC3are = _safe_ac("AATSC", 3, "are")
_AATSC4are = _safe_ac("AATSC", 4, "are")
_AATSC5are = _safe_ac("AATSC", 5, "are")
_AATSC6are = _safe_ac("AATSC", 6, "are")
_AATSC7are = _safe_ac("AATSC", 7, "are")
_AATSC8are = _safe_ac("AATSC", 8, "are")
_AATSC0p = _safe_ac("AATSC", 0, "p")
_AATSC1p = _safe_ac("AATSC", 1, "p")
_AATSC2p = _safe_ac("AATSC", 2, "p")
_AATSC3p = _safe_ac("AATSC", 3, "p")
_AATSC4p = _safe_ac("AATSC", 4, "p")
_AATSC5p = _safe_ac("AATSC", 5, "p")
_AATSC6p = _safe_ac("AATSC", 6, "p")
_AATSC7p = _safe_ac("AATSC", 7, "p")
_AATSC8p = _safe_ac("AATSC", 8, "p")
_AATSC0i = _safe_ac("AATSC", 0, "i")
_AATSC1i = _safe_ac("AATSC", 1, "i")
_AATSC2i = _safe_ac("AATSC", 2, "i")
_AATSC3i = _safe_ac("AATSC", 3, "i")
_AATSC4i = _safe_ac("AATSC", 4, "i")
_AATSC5i = _safe_ac("AATSC", 5, "i")
_AATSC6i = _safe_ac("AATSC", 6, "i")
_AATSC7i = _safe_ac("AATSC", 7, "i")
_AATSC8i = _safe_ac("AATSC", 8, "i")
_MATS1c = _safe_ac("MATS", 1, "c")
_MATS2c = _safe_ac("MATS", 2, "c")
_MATS3c = _safe_ac("MATS", 3, "c")
_MATS4c = _safe_ac("MATS", 4, "c")
_MATS5c = _safe_ac("MATS", 5, "c")
_MATS6c = _safe_ac("MATS", 6, "c")
_MATS7c = _safe_ac("MATS", 7, "c")
_MATS8c = _safe_ac("MATS", 8, "c")
_MATS1dv = _safe_ac("MATS", 1, "dv")
_MATS2dv = _safe_ac("MATS", 2, "dv")
_MATS3dv = _safe_ac("MATS", 3, "dv")
_MATS4dv = _safe_ac("MATS", 4, "dv")
_MATS5dv = _safe_ac("MATS", 5, "dv")
_MATS6dv = _safe_ac("MATS", 6, "dv")
_MATS7dv = _safe_ac("MATS", 7, "dv")
_MATS8dv = _safe_ac("MATS", 8, "dv")
_MATS1d = _safe_ac("MATS", 1, "d")
_MATS2d = _safe_ac("MATS", 2, "d")
_MATS3d = _safe_ac("MATS", 3, "d")
_MATS4d = _safe_ac("MATS", 4, "d")
_MATS5d = _safe_ac("MATS", 5, "d")
_MATS6d = _safe_ac("MATS", 6, "d")
_MATS7d = _safe_ac("MATS", 7, "d")
_MATS8d = _safe_ac("MATS", 8, "d")
_MATS1s = _safe_ac("MATS", 1, "s")
_MATS2s = _safe_ac("MATS", 2, "s")
_MATS3s = _safe_ac("MATS", 3, "s")
_MATS4s = _safe_ac("MATS", 4, "s")
_MATS5s = _safe_ac("MATS", 5, "s")
_MATS6s = _safe_ac("MATS", 6, "s")
_MATS7s = _safe_ac("MATS", 7, "s")
_MATS8s = _safe_ac("MATS", 8, "s")
_MATS1Z = _safe_ac("MATS", 1, "Z")
_MATS2Z = _safe_ac("MATS", 2, "Z")
_MATS3Z = _safe_ac("MATS", 3, "Z")
_MATS4Z = _safe_ac("MATS", 4, "Z")
_MATS5Z = _safe_ac("MATS", 5, "Z")
_MATS6Z = _safe_ac("MATS", 6, "Z")
_MATS7Z = _safe_ac("MATS", 7, "Z")
_MATS8Z = _safe_ac("MATS", 8, "Z")
_MATS1m = _safe_ac("MATS", 1, "m")
_MATS2m = _safe_ac("MATS", 2, "m")
_MATS3m = _safe_ac("MATS", 3, "m")
_MATS4m = _safe_ac("MATS", 4, "m")
_MATS5m = _safe_ac("MATS", 5, "m")
_MATS6m = _safe_ac("MATS", 6, "m")
_MATS7m = _safe_ac("MATS", 7, "m")
_MATS8m = _safe_ac("MATS", 8, "m")
_MATS1v = _safe_ac("MATS", 1, "v")
_MATS2v = _safe_ac("MATS", 2, "v")
_MATS3v = _safe_ac("MATS", 3, "v")
_MATS4v = _safe_ac("MATS", 4, "v")
_MATS5v = _safe_ac("MATS", 5, "v")
_MATS6v = _safe_ac("MATS", 6, "v")
_MATS7v = _safe_ac("MATS", 7, "v")
_MATS8v = _safe_ac("MATS", 8, "v")
_MATS1se = _safe_ac("MATS", 1, "se")
_MATS2se = _safe_ac("MATS", 2, "se")
_MATS3se = _safe_ac("MATS", 3, "se")
_MATS4se = _safe_ac("MATS", 4, "se")
_MATS5se = _safe_ac("MATS", 5, "se")
_MATS6se = _safe_ac("MATS", 6, "se")
_MATS7se = _safe_ac("MATS", 7, "se")
_MATS8se = _safe_ac("MATS", 8, "se")
_MATS1pe = _safe_ac("MATS", 1, "pe")
_MATS2pe = _safe_ac("MATS", 2, "pe")
_MATS3pe = _safe_ac("MATS", 3, "pe")
_MATS4pe = _safe_ac("MATS", 4, "pe")
_MATS5pe = _safe_ac("MATS", 5, "pe")
_MATS6pe = _safe_ac("MATS", 6, "pe")
_MATS7pe = _safe_ac("MATS", 7, "pe")
_MATS8pe = _safe_ac("MATS", 8, "pe")
_MATS1are = _safe_ac("MATS", 1, "are")
_MATS2are = _safe_ac("MATS", 2, "are")
_MATS3are = _safe_ac("MATS", 3, "are")
_MATS4are = _safe_ac("MATS", 4, "are")
_MATS5are = _safe_ac("MATS", 5, "are")
_MATS6are = _safe_ac("MATS", 6, "are")
_MATS7are = _safe_ac("MATS", 7, "are")
_MATS8are = _safe_ac("MATS", 8, "are")
_MATS1p = _safe_ac("MATS", 1, "p")
_MATS2p = _safe_ac("MATS", 2, "p")
_MATS3p = _safe_ac("MATS", 3, "p")
_MATS4p = _safe_ac("MATS", 4, "p")
_MATS5p = _safe_ac("MATS", 5, "p")
_MATS6p = _safe_ac("MATS", 6, "p")
_MATS7p = _safe_ac("MATS", 7, "p")
_MATS8p = _safe_ac("MATS", 8, "p")
_MATS1i = _safe_ac("MATS", 1, "i")
_MATS2i = _safe_ac("MATS", 2, "i")
_MATS3i = _safe_ac("MATS", 3, "i")
_MATS4i = _safe_ac("MATS", 4, "i")
_MATS5i = _safe_ac("MATS", 5, "i")
_MATS6i = _safe_ac("MATS", 6, "i")
_MATS7i = _safe_ac("MATS", 7, "i")
_MATS8i = _safe_ac("MATS", 8, "i")
_GATS1c = _safe_ac("GATS", 1, "c")
_GATS2c = _safe_ac("GATS", 2, "c")
_GATS3c = _safe_ac("GATS", 3, "c")
_GATS4c = _safe_ac("GATS", 4, "c")
_GATS5c = _safe_ac("GATS", 5, "c")
_GATS6c = _safe_ac("GATS", 6, "c")
_GATS7c = _safe_ac("GATS", 7, "c")
_GATS8c = _safe_ac("GATS", 8, "c")
_GATS1dv = _safe_ac("GATS", 1, "dv")
_GATS2dv = _safe_ac("GATS", 2, "dv")
_GATS3dv = _safe_ac("GATS", 3, "dv")
_GATS4dv = _safe_ac("GATS", 4, "dv")
_GATS5dv = _safe_ac("GATS", 5, "dv")
_GATS6dv = _safe_ac("GATS", 6, "dv")
_GATS7dv = _safe_ac("GATS", 7, "dv")
_GATS8dv = _safe_ac("GATS", 8, "dv")
_GATS1d = _safe_ac("GATS", 1, "d")
_GATS2d = _safe_ac("GATS", 2, "d")
_GATS3d = _safe_ac("GATS", 3, "d")
_GATS4d = _safe_ac("GATS", 4, "d")
_GATS5d = _safe_ac("GATS", 5, "d")
_GATS6d = _safe_ac("GATS", 6, "d")
_GATS7d = _safe_ac("GATS", 7, "d")
_GATS8d = _safe_ac("GATS", 8, "d")
_GATS1s = _safe_ac("GATS", 1, "s")
_GATS2s = _safe_ac("GATS", 2, "s")
_GATS3s = _safe_ac("GATS", 3, "s")
_GATS4s = _safe_ac("GATS", 4, "s")
_GATS5s = _safe_ac("GATS", 5, "s")
_GATS6s = _safe_ac("GATS", 6, "s")
_GATS7s = _safe_ac("GATS", 7, "s")
_GATS8s = _safe_ac("GATS", 8, "s")
_GATS1Z = _safe_ac("GATS", 1, "Z")
_GATS2Z = _safe_ac("GATS", 2, "Z")
_GATS3Z = _safe_ac("GATS", 3, "Z")
_GATS4Z = _safe_ac("GATS", 4, "Z")
_GATS5Z = _safe_ac("GATS", 5, "Z")
_GATS6Z = _safe_ac("GATS", 6, "Z")
_GATS7Z = _safe_ac("GATS", 7, "Z")
_GATS8Z = _safe_ac("GATS", 8, "Z")
_GATS1m = _safe_ac("GATS", 1, "m")
_GATS2m = _safe_ac("GATS", 2, "m")
_GATS3m = _safe_ac("GATS", 3, "m")
_GATS4m = _safe_ac("GATS", 4, "m")
_GATS5m = _safe_ac("GATS", 5, "m")
_GATS6m = _safe_ac("GATS", 6, "m")
_GATS7m = _safe_ac("GATS", 7, "m")
_GATS8m = _safe_ac("GATS", 8, "m")
_GATS1v = _safe_ac("GATS", 1, "v")
_GATS2v = _safe_ac("GATS", 2, "v")
_GATS3v = _safe_ac("GATS", 3, "v")
_GATS4v = _safe_ac("GATS", 4, "v")
_GATS5v = _safe_ac("GATS", 5, "v")
_GATS6v = _safe_ac("GATS", 6, "v")
_GATS7v = _safe_ac("GATS", 7, "v")
_GATS8v = _safe_ac("GATS", 8, "v")
_GATS1se = _safe_ac("GATS", 1, "se")
_GATS2se = _safe_ac("GATS", 2, "se")
_GATS3se = _safe_ac("GATS", 3, "se")
_GATS4se = _safe_ac("GATS", 4, "se")
_GATS5se = _safe_ac("GATS", 5, "se")
_GATS6se = _safe_ac("GATS", 6, "se")
_GATS7se = _safe_ac("GATS", 7, "se")
_GATS8se = _safe_ac("GATS", 8, "se")
_GATS1pe = _safe_ac("GATS", 1, "pe")
_GATS2pe = _safe_ac("GATS", 2, "pe")
_GATS3pe = _safe_ac("GATS", 3, "pe")
_GATS4pe = _safe_ac("GATS", 4, "pe")
_GATS5pe = _safe_ac("GATS", 5, "pe")
_GATS6pe = _safe_ac("GATS", 6, "pe")
_GATS7pe = _safe_ac("GATS", 7, "pe")
_GATS8pe = _safe_ac("GATS", 8, "pe")
_GATS1are = _safe_ac("GATS", 1, "are")
_GATS2are = _safe_ac("GATS", 2, "are")
_GATS3are = _safe_ac("GATS", 3, "are")
_GATS4are = _safe_ac("GATS", 4, "are")
_GATS5are = _safe_ac("GATS", 5, "are")
_GATS6are = _safe_ac("GATS", 6, "are")
_GATS7are = _safe_ac("GATS", 7, "are")
_GATS8are = _safe_ac("GATS", 8, "are")
_GATS1p = _safe_ac("GATS", 1, "p")
_GATS2p = _safe_ac("GATS", 2, "p")
_GATS3p = _safe_ac("GATS", 3, "p")
_GATS4p = _safe_ac("GATS", 4, "p")
_GATS5p = _safe_ac("GATS", 5, "p")
_GATS6p = _safe_ac("GATS", 6, "p")
_GATS7p = _safe_ac("GATS", 7, "p")
_GATS8p = _safe_ac("GATS", 8, "p")
_GATS1i = _safe_ac("GATS", 1, "i")
_GATS2i = _safe_ac("GATS", 2, "i")
_GATS3i = _safe_ac("GATS", 3, "i")
_GATS4i = _safe_ac("GATS", 4, "i")
_GATS5i = _safe_ac("GATS", 5, "i")
_GATS6i = _safe_ac("GATS", 6, "i")
_GATS7i = _safe_ac("GATS", 7, "i")
_GATS8i = _safe_ac("GATS", 8, "i")


def ATS0dv(mol):
    """ATS0dv (BrotoMoreau; ATS lag=0 prop=dv)."""
    return _ATS0dv(mol)


def ATS1dv(mol):
    """ATS1dv (BrotoMoreau; ATS lag=1 prop=dv)."""
    return _ATS1dv(mol)


def ATS2dv(mol):
    """ATS2dv (BrotoMoreau; ATS lag=2 prop=dv)."""
    return _ATS2dv(mol)


def ATS3dv(mol):
    """ATS3dv (BrotoMoreau; ATS lag=3 prop=dv)."""
    return _ATS3dv(mol)


def ATS4dv(mol):
    """ATS4dv (BrotoMoreau; ATS lag=4 prop=dv)."""
    return _ATS4dv(mol)


def ATS5dv(mol):
    """ATS5dv (BrotoMoreau; ATS lag=5 prop=dv)."""
    return _ATS5dv(mol)


def ATS6dv(mol):
    """ATS6dv (BrotoMoreau; ATS lag=6 prop=dv)."""
    return _ATS6dv(mol)


def ATS7dv(mol):
    """ATS7dv (BrotoMoreau; ATS lag=7 prop=dv)."""
    return _ATS7dv(mol)


def ATS8dv(mol):
    """ATS8dv (BrotoMoreau; ATS lag=8 prop=dv)."""
    return _ATS8dv(mol)


def ATS0d(mol):
    """ATS0d (BrotoMoreau; ATS lag=0 prop=d)."""
    return _ATS0d(mol)


def ATS1d(mol):
    """ATS1d (BrotoMoreau; ATS lag=1 prop=d)."""
    return _ATS1d(mol)


def ATS2d(mol):
    """ATS2d (BrotoMoreau; ATS lag=2 prop=d)."""
    return _ATS2d(mol)


def ATS3d(mol):
    """ATS3d (BrotoMoreau; ATS lag=3 prop=d)."""
    return _ATS3d(mol)


def ATS4d(mol):
    """ATS4d (BrotoMoreau; ATS lag=4 prop=d)."""
    return _ATS4d(mol)


def ATS5d(mol):
    """ATS5d (BrotoMoreau; ATS lag=5 prop=d)."""
    return _ATS5d(mol)


def ATS6d(mol):
    """ATS6d (BrotoMoreau; ATS lag=6 prop=d)."""
    return _ATS6d(mol)


def ATS7d(mol):
    """ATS7d (BrotoMoreau; ATS lag=7 prop=d)."""
    return _ATS7d(mol)


def ATS8d(mol):
    """ATS8d (BrotoMoreau; ATS lag=8 prop=d)."""
    return _ATS8d(mol)


def ATS0s(mol):
    """ATS0s (BrotoMoreau; ATS lag=0 prop=s)."""
    return _ATS0s(mol)


def ATS1s(mol):
    """ATS1s (BrotoMoreau; ATS lag=1 prop=s)."""
    return _ATS1s(mol)


def ATS2s(mol):
    """ATS2s (BrotoMoreau; ATS lag=2 prop=s)."""
    return _ATS2s(mol)


def ATS3s(mol):
    """ATS3s (BrotoMoreau; ATS lag=3 prop=s)."""
    return _ATS3s(mol)


def ATS4s(mol):
    """ATS4s (BrotoMoreau; ATS lag=4 prop=s)."""
    return _ATS4s(mol)


def ATS5s(mol):
    """ATS5s (BrotoMoreau; ATS lag=5 prop=s)."""
    return _ATS5s(mol)


def ATS6s(mol):
    """ATS6s (BrotoMoreau; ATS lag=6 prop=s)."""
    return _ATS6s(mol)


def ATS7s(mol):
    """ATS7s (BrotoMoreau; ATS lag=7 prop=s)."""
    return _ATS7s(mol)


def ATS8s(mol):
    """ATS8s (BrotoMoreau; ATS lag=8 prop=s)."""
    return _ATS8s(mol)


def ATS0Z(mol):
    """ATS0Z (BrotoMoreau; ATS lag=0 prop=Z)."""
    return _ATS0Z(mol)


def ATS1Z(mol):
    """ATS1Z (BrotoMoreau; ATS lag=1 prop=Z)."""
    return _ATS1Z(mol)


def ATS2Z(mol):
    """ATS2Z (BrotoMoreau; ATS lag=2 prop=Z)."""
    return _ATS2Z(mol)


def ATS3Z(mol):
    """ATS3Z (BrotoMoreau; ATS lag=3 prop=Z)."""
    return _ATS3Z(mol)


def ATS4Z(mol):
    """ATS4Z (BrotoMoreau; ATS lag=4 prop=Z)."""
    return _ATS4Z(mol)


def ATS5Z(mol):
    """ATS5Z (BrotoMoreau; ATS lag=5 prop=Z)."""
    return _ATS5Z(mol)


def ATS6Z(mol):
    """ATS6Z (BrotoMoreau; ATS lag=6 prop=Z)."""
    return _ATS6Z(mol)


def ATS7Z(mol):
    """ATS7Z (BrotoMoreau; ATS lag=7 prop=Z)."""
    return _ATS7Z(mol)


def ATS8Z(mol):
    """ATS8Z (BrotoMoreau; ATS lag=8 prop=Z)."""
    return _ATS8Z(mol)


def ATS0m(mol):
    """ATS0m (BrotoMoreau; ATS lag=0 prop=m)."""
    return _ATS0m(mol)


def ATS1m(mol):
    """ATS1m (BrotoMoreau; ATS lag=1 prop=m)."""
    return _ATS1m(mol)


def ATS2m(mol):
    """ATS2m (BrotoMoreau; ATS lag=2 prop=m)."""
    return _ATS2m(mol)


def ATS3m(mol):
    """ATS3m (BrotoMoreau; ATS lag=3 prop=m)."""
    return _ATS3m(mol)


def ATS4m(mol):
    """ATS4m (BrotoMoreau; ATS lag=4 prop=m)."""
    return _ATS4m(mol)


def ATS5m(mol):
    """ATS5m (BrotoMoreau; ATS lag=5 prop=m)."""
    return _ATS5m(mol)


def ATS6m(mol):
    """ATS6m (BrotoMoreau; ATS lag=6 prop=m)."""
    return _ATS6m(mol)


def ATS7m(mol):
    """ATS7m (BrotoMoreau; ATS lag=7 prop=m)."""
    return _ATS7m(mol)


def ATS8m(mol):
    """ATS8m (BrotoMoreau; ATS lag=8 prop=m)."""
    return _ATS8m(mol)


def ATS0v(mol):
    """ATS0v (BrotoMoreau; ATS lag=0 prop=v)."""
    return _ATS0v(mol)


def ATS1v(mol):
    """ATS1v (BrotoMoreau; ATS lag=1 prop=v)."""
    return _ATS1v(mol)


def ATS2v(mol):
    """ATS2v (BrotoMoreau; ATS lag=2 prop=v)."""
    return _ATS2v(mol)


def ATS3v(mol):
    """ATS3v (BrotoMoreau; ATS lag=3 prop=v)."""
    return _ATS3v(mol)


def ATS4v(mol):
    """ATS4v (BrotoMoreau; ATS lag=4 prop=v)."""
    return _ATS4v(mol)


def ATS5v(mol):
    """ATS5v (BrotoMoreau; ATS lag=5 prop=v)."""
    return _ATS5v(mol)


def ATS6v(mol):
    """ATS6v (BrotoMoreau; ATS lag=6 prop=v)."""
    return _ATS6v(mol)


def ATS7v(mol):
    """ATS7v (BrotoMoreau; ATS lag=7 prop=v)."""
    return _ATS7v(mol)


def ATS8v(mol):
    """ATS8v (BrotoMoreau; ATS lag=8 prop=v)."""
    return _ATS8v(mol)


def ATS0se(mol):
    """ATS0se (BrotoMoreau; ATS lag=0 prop=se)."""
    return _ATS0se(mol)


def ATS1se(mol):
    """ATS1se (BrotoMoreau; ATS lag=1 prop=se)."""
    return _ATS1se(mol)


def ATS2se(mol):
    """ATS2se (BrotoMoreau; ATS lag=2 prop=se)."""
    return _ATS2se(mol)


def ATS3se(mol):
    """ATS3se (BrotoMoreau; ATS lag=3 prop=se)."""
    return _ATS3se(mol)


def ATS4se(mol):
    """ATS4se (BrotoMoreau; ATS lag=4 prop=se)."""
    return _ATS4se(mol)


def ATS5se(mol):
    """ATS5se (BrotoMoreau; ATS lag=5 prop=se)."""
    return _ATS5se(mol)


def ATS6se(mol):
    """ATS6se (BrotoMoreau; ATS lag=6 prop=se)."""
    return _ATS6se(mol)


def ATS7se(mol):
    """ATS7se (BrotoMoreau; ATS lag=7 prop=se)."""
    return _ATS7se(mol)


def ATS8se(mol):
    """ATS8se (BrotoMoreau; ATS lag=8 prop=se)."""
    return _ATS8se(mol)


def ATS0pe(mol):
    """ATS0pe (BrotoMoreau; ATS lag=0 prop=pe)."""
    return _ATS0pe(mol)


def ATS1pe(mol):
    """ATS1pe (BrotoMoreau; ATS lag=1 prop=pe)."""
    return _ATS1pe(mol)


def ATS2pe(mol):
    """ATS2pe (BrotoMoreau; ATS lag=2 prop=pe)."""
    return _ATS2pe(mol)


def ATS3pe(mol):
    """ATS3pe (BrotoMoreau; ATS lag=3 prop=pe)."""
    return _ATS3pe(mol)


def ATS4pe(mol):
    """ATS4pe (BrotoMoreau; ATS lag=4 prop=pe)."""
    return _ATS4pe(mol)


def ATS5pe(mol):
    """ATS5pe (BrotoMoreau; ATS lag=5 prop=pe)."""
    return _ATS5pe(mol)


def ATS6pe(mol):
    """ATS6pe (BrotoMoreau; ATS lag=6 prop=pe)."""
    return _ATS6pe(mol)


def ATS7pe(mol):
    """ATS7pe (BrotoMoreau; ATS lag=7 prop=pe)."""
    return _ATS7pe(mol)


def ATS8pe(mol):
    """ATS8pe (BrotoMoreau; ATS lag=8 prop=pe)."""
    return _ATS8pe(mol)


def ATS0are(mol):
    """ATS0are (BrotoMoreau; ATS lag=0 prop=are)."""
    return _ATS0are(mol)


def ATS1are(mol):
    """ATS1are (BrotoMoreau; ATS lag=1 prop=are)."""
    return _ATS1are(mol)


def ATS2are(mol):
    """ATS2are (BrotoMoreau; ATS lag=2 prop=are)."""
    return _ATS2are(mol)


def ATS3are(mol):
    """ATS3are (BrotoMoreau; ATS lag=3 prop=are)."""
    return _ATS3are(mol)


def ATS4are(mol):
    """ATS4are (BrotoMoreau; ATS lag=4 prop=are)."""
    return _ATS4are(mol)


def ATS5are(mol):
    """ATS5are (BrotoMoreau; ATS lag=5 prop=are)."""
    return _ATS5are(mol)


def ATS6are(mol):
    """ATS6are (BrotoMoreau; ATS lag=6 prop=are)."""
    return _ATS6are(mol)


def ATS7are(mol):
    """ATS7are (BrotoMoreau; ATS lag=7 prop=are)."""
    return _ATS7are(mol)


def ATS8are(mol):
    """ATS8are (BrotoMoreau; ATS lag=8 prop=are)."""
    return _ATS8are(mol)


def ATS0p(mol):
    """ATS0p (BrotoMoreau; ATS lag=0 prop=p)."""
    return _ATS0p(mol)


def ATS1p(mol):
    """ATS1p (BrotoMoreau; ATS lag=1 prop=p)."""
    return _ATS1p(mol)


def ATS2p(mol):
    """ATS2p (BrotoMoreau; ATS lag=2 prop=p)."""
    return _ATS2p(mol)


def ATS3p(mol):
    """ATS3p (BrotoMoreau; ATS lag=3 prop=p)."""
    return _ATS3p(mol)


def ATS4p(mol):
    """ATS4p (BrotoMoreau; ATS lag=4 prop=p)."""
    return _ATS4p(mol)


def ATS5p(mol):
    """ATS5p (BrotoMoreau; ATS lag=5 prop=p)."""
    return _ATS5p(mol)


def ATS6p(mol):
    """ATS6p (BrotoMoreau; ATS lag=6 prop=p)."""
    return _ATS6p(mol)


def ATS7p(mol):
    """ATS7p (BrotoMoreau; ATS lag=7 prop=p)."""
    return _ATS7p(mol)


def ATS8p(mol):
    """ATS8p (BrotoMoreau; ATS lag=8 prop=p)."""
    return _ATS8p(mol)


def ATS0i(mol):
    """ATS0i (BrotoMoreau; ATS lag=0 prop=i)."""
    return _ATS0i(mol)


def ATS1i(mol):
    """ATS1i (BrotoMoreau; ATS lag=1 prop=i)."""
    return _ATS1i(mol)


def ATS2i(mol):
    """ATS2i (BrotoMoreau; ATS lag=2 prop=i)."""
    return _ATS2i(mol)


def ATS3i(mol):
    """ATS3i (BrotoMoreau; ATS lag=3 prop=i)."""
    return _ATS3i(mol)


def ATS4i(mol):
    """ATS4i (BrotoMoreau; ATS lag=4 prop=i)."""
    return _ATS4i(mol)


def ATS5i(mol):
    """ATS5i (BrotoMoreau; ATS lag=5 prop=i)."""
    return _ATS5i(mol)


def ATS6i(mol):
    """ATS6i (BrotoMoreau; ATS lag=6 prop=i)."""
    return _ATS6i(mol)


def ATS7i(mol):
    """ATS7i (BrotoMoreau; ATS lag=7 prop=i)."""
    return _ATS7i(mol)


def ATS8i(mol):
    """ATS8i (BrotoMoreau; ATS lag=8 prop=i)."""
    return _ATS8i(mol)


def AATS0dv(mol):
    """AATS0dv (BrotoMoreau; AATS lag=0 prop=dv)."""
    return _AATS0dv(mol)


def AATS1dv(mol):
    """AATS1dv (BrotoMoreau; AATS lag=1 prop=dv)."""
    return _AATS1dv(mol)


def AATS2dv(mol):
    """AATS2dv (BrotoMoreau; AATS lag=2 prop=dv)."""
    return _AATS2dv(mol)


def AATS3dv(mol):
    """AATS3dv (BrotoMoreau; AATS lag=3 prop=dv)."""
    return _AATS3dv(mol)


def AATS4dv(mol):
    """AATS4dv (BrotoMoreau; AATS lag=4 prop=dv)."""
    return _AATS4dv(mol)


def AATS5dv(mol):
    """AATS5dv (BrotoMoreau; AATS lag=5 prop=dv)."""
    return _AATS5dv(mol)


def AATS6dv(mol):
    """AATS6dv (BrotoMoreau; AATS lag=6 prop=dv)."""
    return _AATS6dv(mol)


def AATS7dv(mol):
    """AATS7dv (BrotoMoreau; AATS lag=7 prop=dv)."""
    return _AATS7dv(mol)


def AATS8dv(mol):
    """AATS8dv (BrotoMoreau; AATS lag=8 prop=dv)."""
    return _AATS8dv(mol)


def AATS0d(mol):
    """AATS0d (BrotoMoreau; AATS lag=0 prop=d)."""
    return _AATS0d(mol)


def AATS1d(mol):
    """AATS1d (BrotoMoreau; AATS lag=1 prop=d)."""
    return _AATS1d(mol)


def AATS2d(mol):
    """AATS2d (BrotoMoreau; AATS lag=2 prop=d)."""
    return _AATS2d(mol)


def AATS3d(mol):
    """AATS3d (BrotoMoreau; AATS lag=3 prop=d)."""
    return _AATS3d(mol)


def AATS4d(mol):
    """AATS4d (BrotoMoreau; AATS lag=4 prop=d)."""
    return _AATS4d(mol)


def AATS5d(mol):
    """AATS5d (BrotoMoreau; AATS lag=5 prop=d)."""
    return _AATS5d(mol)


def AATS6d(mol):
    """AATS6d (BrotoMoreau; AATS lag=6 prop=d)."""
    return _AATS6d(mol)


def AATS7d(mol):
    """AATS7d (BrotoMoreau; AATS lag=7 prop=d)."""
    return _AATS7d(mol)


def AATS8d(mol):
    """AATS8d (BrotoMoreau; AATS lag=8 prop=d)."""
    return _AATS8d(mol)


def AATS0s(mol):
    """AATS0s (BrotoMoreau; AATS lag=0 prop=s)."""
    return _AATS0s(mol)


def AATS1s(mol):
    """AATS1s (BrotoMoreau; AATS lag=1 prop=s)."""
    return _AATS1s(mol)


def AATS2s(mol):
    """AATS2s (BrotoMoreau; AATS lag=2 prop=s)."""
    return _AATS2s(mol)


def AATS3s(mol):
    """AATS3s (BrotoMoreau; AATS lag=3 prop=s)."""
    return _AATS3s(mol)


def AATS4s(mol):
    """AATS4s (BrotoMoreau; AATS lag=4 prop=s)."""
    return _AATS4s(mol)


def AATS5s(mol):
    """AATS5s (BrotoMoreau; AATS lag=5 prop=s)."""
    return _AATS5s(mol)


def AATS6s(mol):
    """AATS6s (BrotoMoreau; AATS lag=6 prop=s)."""
    return _AATS6s(mol)


def AATS7s(mol):
    """AATS7s (BrotoMoreau; AATS lag=7 prop=s)."""
    return _AATS7s(mol)


def AATS8s(mol):
    """AATS8s (BrotoMoreau; AATS lag=8 prop=s)."""
    return _AATS8s(mol)


def AATS0Z(mol):
    """AATS0Z (BrotoMoreau; AATS lag=0 prop=Z)."""
    return _AATS0Z(mol)


def AATS1Z(mol):
    """AATS1Z (BrotoMoreau; AATS lag=1 prop=Z)."""
    return _AATS1Z(mol)


def AATS2Z(mol):
    """AATS2Z (BrotoMoreau; AATS lag=2 prop=Z)."""
    return _AATS2Z(mol)


def AATS3Z(mol):
    """AATS3Z (BrotoMoreau; AATS lag=3 prop=Z)."""
    return _AATS3Z(mol)


def AATS4Z(mol):
    """AATS4Z (BrotoMoreau; AATS lag=4 prop=Z)."""
    return _AATS4Z(mol)


def AATS5Z(mol):
    """AATS5Z (BrotoMoreau; AATS lag=5 prop=Z)."""
    return _AATS5Z(mol)


def AATS6Z(mol):
    """AATS6Z (BrotoMoreau; AATS lag=6 prop=Z)."""
    return _AATS6Z(mol)


def AATS7Z(mol):
    """AATS7Z (BrotoMoreau; AATS lag=7 prop=Z)."""
    return _AATS7Z(mol)


def AATS8Z(mol):
    """AATS8Z (BrotoMoreau; AATS lag=8 prop=Z)."""
    return _AATS8Z(mol)


def AATS0m(mol):
    """AATS0m (BrotoMoreau; AATS lag=0 prop=m)."""
    return _AATS0m(mol)


def AATS1m(mol):
    """AATS1m (BrotoMoreau; AATS lag=1 prop=m)."""
    return _AATS1m(mol)


def AATS2m(mol):
    """AATS2m (BrotoMoreau; AATS lag=2 prop=m)."""
    return _AATS2m(mol)


def AATS3m(mol):
    """AATS3m (BrotoMoreau; AATS lag=3 prop=m)."""
    return _AATS3m(mol)


def AATS4m(mol):
    """AATS4m (BrotoMoreau; AATS lag=4 prop=m)."""
    return _AATS4m(mol)


def AATS5m(mol):
    """AATS5m (BrotoMoreau; AATS lag=5 prop=m)."""
    return _AATS5m(mol)


def AATS6m(mol):
    """AATS6m (BrotoMoreau; AATS lag=6 prop=m)."""
    return _AATS6m(mol)


def AATS7m(mol):
    """AATS7m (BrotoMoreau; AATS lag=7 prop=m)."""
    return _AATS7m(mol)


def AATS8m(mol):
    """AATS8m (BrotoMoreau; AATS lag=8 prop=m)."""
    return _AATS8m(mol)


def AATS0v(mol):
    """AATS0v (BrotoMoreau; AATS lag=0 prop=v)."""
    return _AATS0v(mol)


def AATS1v(mol):
    """AATS1v (BrotoMoreau; AATS lag=1 prop=v)."""
    return _AATS1v(mol)


def AATS2v(mol):
    """AATS2v (BrotoMoreau; AATS lag=2 prop=v)."""
    return _AATS2v(mol)


def AATS3v(mol):
    """AATS3v (BrotoMoreau; AATS lag=3 prop=v)."""
    return _AATS3v(mol)


def AATS4v(mol):
    """AATS4v (BrotoMoreau; AATS lag=4 prop=v)."""
    return _AATS4v(mol)


def AATS5v(mol):
    """AATS5v (BrotoMoreau; AATS lag=5 prop=v)."""
    return _AATS5v(mol)


def AATS6v(mol):
    """AATS6v (BrotoMoreau; AATS lag=6 prop=v)."""
    return _AATS6v(mol)


def AATS7v(mol):
    """AATS7v (BrotoMoreau; AATS lag=7 prop=v)."""
    return _AATS7v(mol)


def AATS8v(mol):
    """AATS8v (BrotoMoreau; AATS lag=8 prop=v)."""
    return _AATS8v(mol)


def AATS0se(mol):
    """AATS0se (BrotoMoreau; AATS lag=0 prop=se)."""
    return _AATS0se(mol)


def AATS1se(mol):
    """AATS1se (BrotoMoreau; AATS lag=1 prop=se)."""
    return _AATS1se(mol)


def AATS2se(mol):
    """AATS2se (BrotoMoreau; AATS lag=2 prop=se)."""
    return _AATS2se(mol)


def AATS3se(mol):
    """AATS3se (BrotoMoreau; AATS lag=3 prop=se)."""
    return _AATS3se(mol)


def AATS4se(mol):
    """AATS4se (BrotoMoreau; AATS lag=4 prop=se)."""
    return _AATS4se(mol)


def AATS5se(mol):
    """AATS5se (BrotoMoreau; AATS lag=5 prop=se)."""
    return _AATS5se(mol)


def AATS6se(mol):
    """AATS6se (BrotoMoreau; AATS lag=6 prop=se)."""
    return _AATS6se(mol)


def AATS7se(mol):
    """AATS7se (BrotoMoreau; AATS lag=7 prop=se)."""
    return _AATS7se(mol)


def AATS8se(mol):
    """AATS8se (BrotoMoreau; AATS lag=8 prop=se)."""
    return _AATS8se(mol)


def AATS0pe(mol):
    """AATS0pe (BrotoMoreau; AATS lag=0 prop=pe)."""
    return _AATS0pe(mol)


def AATS1pe(mol):
    """AATS1pe (BrotoMoreau; AATS lag=1 prop=pe)."""
    return _AATS1pe(mol)


def AATS2pe(mol):
    """AATS2pe (BrotoMoreau; AATS lag=2 prop=pe)."""
    return _AATS2pe(mol)


def AATS3pe(mol):
    """AATS3pe (BrotoMoreau; AATS lag=3 prop=pe)."""
    return _AATS3pe(mol)


def AATS4pe(mol):
    """AATS4pe (BrotoMoreau; AATS lag=4 prop=pe)."""
    return _AATS4pe(mol)


def AATS5pe(mol):
    """AATS5pe (BrotoMoreau; AATS lag=5 prop=pe)."""
    return _AATS5pe(mol)


def AATS6pe(mol):
    """AATS6pe (BrotoMoreau; AATS lag=6 prop=pe)."""
    return _AATS6pe(mol)


def AATS7pe(mol):
    """AATS7pe (BrotoMoreau; AATS lag=7 prop=pe)."""
    return _AATS7pe(mol)


def AATS8pe(mol):
    """AATS8pe (BrotoMoreau; AATS lag=8 prop=pe)."""
    return _AATS8pe(mol)


def AATS0are(mol):
    """AATS0are (BrotoMoreau; AATS lag=0 prop=are)."""
    return _AATS0are(mol)


def AATS1are(mol):
    """AATS1are (BrotoMoreau; AATS lag=1 prop=are)."""
    return _AATS1are(mol)


def AATS2are(mol):
    """AATS2are (BrotoMoreau; AATS lag=2 prop=are)."""
    return _AATS2are(mol)


def AATS3are(mol):
    """AATS3are (BrotoMoreau; AATS lag=3 prop=are)."""
    return _AATS3are(mol)


def AATS4are(mol):
    """AATS4are (BrotoMoreau; AATS lag=4 prop=are)."""
    return _AATS4are(mol)


def AATS5are(mol):
    """AATS5are (BrotoMoreau; AATS lag=5 prop=are)."""
    return _AATS5are(mol)


def AATS6are(mol):
    """AATS6are (BrotoMoreau; AATS lag=6 prop=are)."""
    return _AATS6are(mol)


def AATS7are(mol):
    """AATS7are (BrotoMoreau; AATS lag=7 prop=are)."""
    return _AATS7are(mol)


def AATS8are(mol):
    """AATS8are (BrotoMoreau; AATS lag=8 prop=are)."""
    return _AATS8are(mol)


def AATS0p(mol):
    """AATS0p (BrotoMoreau; AATS lag=0 prop=p)."""
    return _AATS0p(mol)


def AATS1p(mol):
    """AATS1p (BrotoMoreau; AATS lag=1 prop=p)."""
    return _AATS1p(mol)


def AATS2p(mol):
    """AATS2p (BrotoMoreau; AATS lag=2 prop=p)."""
    return _AATS2p(mol)


def AATS3p(mol):
    """AATS3p (BrotoMoreau; AATS lag=3 prop=p)."""
    return _AATS3p(mol)


def AATS4p(mol):
    """AATS4p (BrotoMoreau; AATS lag=4 prop=p)."""
    return _AATS4p(mol)


def AATS5p(mol):
    """AATS5p (BrotoMoreau; AATS lag=5 prop=p)."""
    return _AATS5p(mol)


def AATS6p(mol):
    """AATS6p (BrotoMoreau; AATS lag=6 prop=p)."""
    return _AATS6p(mol)


def AATS7p(mol):
    """AATS7p (BrotoMoreau; AATS lag=7 prop=p)."""
    return _AATS7p(mol)


def AATS8p(mol):
    """AATS8p (BrotoMoreau; AATS lag=8 prop=p)."""
    return _AATS8p(mol)


def AATS0i(mol):
    """AATS0i (BrotoMoreau; AATS lag=0 prop=i)."""
    return _AATS0i(mol)


def AATS1i(mol):
    """AATS1i (BrotoMoreau; AATS lag=1 prop=i)."""
    return _AATS1i(mol)


def AATS2i(mol):
    """AATS2i (BrotoMoreau; AATS lag=2 prop=i)."""
    return _AATS2i(mol)


def AATS3i(mol):
    """AATS3i (BrotoMoreau; AATS lag=3 prop=i)."""
    return _AATS3i(mol)


def AATS4i(mol):
    """AATS4i (BrotoMoreau; AATS lag=4 prop=i)."""
    return _AATS4i(mol)


def AATS5i(mol):
    """AATS5i (BrotoMoreau; AATS lag=5 prop=i)."""
    return _AATS5i(mol)


def AATS6i(mol):
    """AATS6i (BrotoMoreau; AATS lag=6 prop=i)."""
    return _AATS6i(mol)


def AATS7i(mol):
    """AATS7i (BrotoMoreau; AATS lag=7 prop=i)."""
    return _AATS7i(mol)


def AATS8i(mol):
    """AATS8i (BrotoMoreau; AATS lag=8 prop=i)."""
    return _AATS8i(mol)


def ATSC0c(mol):
    """ATSC0c (BrotoMoreau; ATSC lag=0 prop=c)."""
    return _ATSC0c(mol)


def ATSC1c(mol):
    """ATSC1c (BrotoMoreau; ATSC lag=1 prop=c)."""
    return _ATSC1c(mol)


def ATSC2c(mol):
    """ATSC2c (BrotoMoreau; ATSC lag=2 prop=c)."""
    return _ATSC2c(mol)


def ATSC3c(mol):
    """ATSC3c (BrotoMoreau; ATSC lag=3 prop=c)."""
    return _ATSC3c(mol)


def ATSC4c(mol):
    """ATSC4c (BrotoMoreau; ATSC lag=4 prop=c)."""
    return _ATSC4c(mol)


def ATSC5c(mol):
    """ATSC5c (BrotoMoreau; ATSC lag=5 prop=c)."""
    return _ATSC5c(mol)


def ATSC6c(mol):
    """ATSC6c (BrotoMoreau; ATSC lag=6 prop=c)."""
    return _ATSC6c(mol)


def ATSC7c(mol):
    """ATSC7c (BrotoMoreau; ATSC lag=7 prop=c)."""
    return _ATSC7c(mol)


def ATSC8c(mol):
    """ATSC8c (BrotoMoreau; ATSC lag=8 prop=c)."""
    return _ATSC8c(mol)


def ATSC0dv(mol):
    """ATSC0dv (BrotoMoreau; ATSC lag=0 prop=dv)."""
    return _ATSC0dv(mol)


def ATSC1dv(mol):
    """ATSC1dv (BrotoMoreau; ATSC lag=1 prop=dv)."""
    return _ATSC1dv(mol)


def ATSC2dv(mol):
    """ATSC2dv (BrotoMoreau; ATSC lag=2 prop=dv)."""
    return _ATSC2dv(mol)


def ATSC3dv(mol):
    """ATSC3dv (BrotoMoreau; ATSC lag=3 prop=dv)."""
    return _ATSC3dv(mol)


def ATSC4dv(mol):
    """ATSC4dv (BrotoMoreau; ATSC lag=4 prop=dv)."""
    return _ATSC4dv(mol)


def ATSC5dv(mol):
    """ATSC5dv (BrotoMoreau; ATSC lag=5 prop=dv)."""
    return _ATSC5dv(mol)


def ATSC6dv(mol):
    """ATSC6dv (BrotoMoreau; ATSC lag=6 prop=dv)."""
    return _ATSC6dv(mol)


def ATSC7dv(mol):
    """ATSC7dv (BrotoMoreau; ATSC lag=7 prop=dv)."""
    return _ATSC7dv(mol)


def ATSC8dv(mol):
    """ATSC8dv (BrotoMoreau; ATSC lag=8 prop=dv)."""
    return _ATSC8dv(mol)


def ATSC0d(mol):
    """ATSC0d (BrotoMoreau; ATSC lag=0 prop=d)."""
    return _ATSC0d(mol)


def ATSC1d(mol):
    """ATSC1d (BrotoMoreau; ATSC lag=1 prop=d)."""
    return _ATSC1d(mol)


def ATSC2d(mol):
    """ATSC2d (BrotoMoreau; ATSC lag=2 prop=d)."""
    return _ATSC2d(mol)


def ATSC3d(mol):
    """ATSC3d (BrotoMoreau; ATSC lag=3 prop=d)."""
    return _ATSC3d(mol)


def ATSC4d(mol):
    """ATSC4d (BrotoMoreau; ATSC lag=4 prop=d)."""
    return _ATSC4d(mol)


def ATSC5d(mol):
    """ATSC5d (BrotoMoreau; ATSC lag=5 prop=d)."""
    return _ATSC5d(mol)


def ATSC6d(mol):
    """ATSC6d (BrotoMoreau; ATSC lag=6 prop=d)."""
    return _ATSC6d(mol)


def ATSC7d(mol):
    """ATSC7d (BrotoMoreau; ATSC lag=7 prop=d)."""
    return _ATSC7d(mol)


def ATSC8d(mol):
    """ATSC8d (BrotoMoreau; ATSC lag=8 prop=d)."""
    return _ATSC8d(mol)


def ATSC0s(mol):
    """ATSC0s (BrotoMoreau; ATSC lag=0 prop=s)."""
    return _ATSC0s(mol)


def ATSC1s(mol):
    """ATSC1s (BrotoMoreau; ATSC lag=1 prop=s)."""
    return _ATSC1s(mol)


def ATSC2s(mol):
    """ATSC2s (BrotoMoreau; ATSC lag=2 prop=s)."""
    return _ATSC2s(mol)


def ATSC3s(mol):
    """ATSC3s (BrotoMoreau; ATSC lag=3 prop=s)."""
    return _ATSC3s(mol)


def ATSC4s(mol):
    """ATSC4s (BrotoMoreau; ATSC lag=4 prop=s)."""
    return _ATSC4s(mol)


def ATSC5s(mol):
    """ATSC5s (BrotoMoreau; ATSC lag=5 prop=s)."""
    return _ATSC5s(mol)


def ATSC6s(mol):
    """ATSC6s (BrotoMoreau; ATSC lag=6 prop=s)."""
    return _ATSC6s(mol)


def ATSC7s(mol):
    """ATSC7s (BrotoMoreau; ATSC lag=7 prop=s)."""
    return _ATSC7s(mol)


def ATSC8s(mol):
    """ATSC8s (BrotoMoreau; ATSC lag=8 prop=s)."""
    return _ATSC8s(mol)


def ATSC0Z(mol):
    """ATSC0Z (BrotoMoreau; ATSC lag=0 prop=Z)."""
    return _ATSC0Z(mol)


def ATSC1Z(mol):
    """ATSC1Z (BrotoMoreau; ATSC lag=1 prop=Z)."""
    return _ATSC1Z(mol)


def ATSC2Z(mol):
    """ATSC2Z (BrotoMoreau; ATSC lag=2 prop=Z)."""
    return _ATSC2Z(mol)


def ATSC3Z(mol):
    """ATSC3Z (BrotoMoreau; ATSC lag=3 prop=Z)."""
    return _ATSC3Z(mol)


def ATSC4Z(mol):
    """ATSC4Z (BrotoMoreau; ATSC lag=4 prop=Z)."""
    return _ATSC4Z(mol)


def ATSC5Z(mol):
    """ATSC5Z (BrotoMoreau; ATSC lag=5 prop=Z)."""
    return _ATSC5Z(mol)


def ATSC6Z(mol):
    """ATSC6Z (BrotoMoreau; ATSC lag=6 prop=Z)."""
    return _ATSC6Z(mol)


def ATSC7Z(mol):
    """ATSC7Z (BrotoMoreau; ATSC lag=7 prop=Z)."""
    return _ATSC7Z(mol)


def ATSC8Z(mol):
    """ATSC8Z (BrotoMoreau; ATSC lag=8 prop=Z)."""
    return _ATSC8Z(mol)


def ATSC0m(mol):
    """ATSC0m (BrotoMoreau; ATSC lag=0 prop=m)."""
    return _ATSC0m(mol)


def ATSC1m(mol):
    """ATSC1m (BrotoMoreau; ATSC lag=1 prop=m)."""
    return _ATSC1m(mol)


def ATSC2m(mol):
    """ATSC2m (BrotoMoreau; ATSC lag=2 prop=m)."""
    return _ATSC2m(mol)


def ATSC3m(mol):
    """ATSC3m (BrotoMoreau; ATSC lag=3 prop=m)."""
    return _ATSC3m(mol)


def ATSC4m(mol):
    """ATSC4m (BrotoMoreau; ATSC lag=4 prop=m)."""
    return _ATSC4m(mol)


def ATSC5m(mol):
    """ATSC5m (BrotoMoreau; ATSC lag=5 prop=m)."""
    return _ATSC5m(mol)


def ATSC6m(mol):
    """ATSC6m (BrotoMoreau; ATSC lag=6 prop=m)."""
    return _ATSC6m(mol)


def ATSC7m(mol):
    """ATSC7m (BrotoMoreau; ATSC lag=7 prop=m)."""
    return _ATSC7m(mol)


def ATSC8m(mol):
    """ATSC8m (BrotoMoreau; ATSC lag=8 prop=m)."""
    return _ATSC8m(mol)


def ATSC0v(mol):
    """ATSC0v (BrotoMoreau; ATSC lag=0 prop=v)."""
    return _ATSC0v(mol)


def ATSC1v(mol):
    """ATSC1v (BrotoMoreau; ATSC lag=1 prop=v)."""
    return _ATSC1v(mol)


def ATSC2v(mol):
    """ATSC2v (BrotoMoreau; ATSC lag=2 prop=v)."""
    return _ATSC2v(mol)


def ATSC3v(mol):
    """ATSC3v (BrotoMoreau; ATSC lag=3 prop=v)."""
    return _ATSC3v(mol)


def ATSC4v(mol):
    """ATSC4v (BrotoMoreau; ATSC lag=4 prop=v)."""
    return _ATSC4v(mol)


def ATSC5v(mol):
    """ATSC5v (BrotoMoreau; ATSC lag=5 prop=v)."""
    return _ATSC5v(mol)


def ATSC6v(mol):
    """ATSC6v (BrotoMoreau; ATSC lag=6 prop=v)."""
    return _ATSC6v(mol)


def ATSC7v(mol):
    """ATSC7v (BrotoMoreau; ATSC lag=7 prop=v)."""
    return _ATSC7v(mol)


def ATSC8v(mol):
    """ATSC8v (BrotoMoreau; ATSC lag=8 prop=v)."""
    return _ATSC8v(mol)


def ATSC0se(mol):
    """ATSC0se (BrotoMoreau; ATSC lag=0 prop=se)."""
    return _ATSC0se(mol)


def ATSC1se(mol):
    """ATSC1se (BrotoMoreau; ATSC lag=1 prop=se)."""
    return _ATSC1se(mol)


def ATSC2se(mol):
    """ATSC2se (BrotoMoreau; ATSC lag=2 prop=se)."""
    return _ATSC2se(mol)


def ATSC3se(mol):
    """ATSC3se (BrotoMoreau; ATSC lag=3 prop=se)."""
    return _ATSC3se(mol)


def ATSC4se(mol):
    """ATSC4se (BrotoMoreau; ATSC lag=4 prop=se)."""
    return _ATSC4se(mol)


def ATSC5se(mol):
    """ATSC5se (BrotoMoreau; ATSC lag=5 prop=se)."""
    return _ATSC5se(mol)


def ATSC6se(mol):
    """ATSC6se (BrotoMoreau; ATSC lag=6 prop=se)."""
    return _ATSC6se(mol)


def ATSC7se(mol):
    """ATSC7se (BrotoMoreau; ATSC lag=7 prop=se)."""
    return _ATSC7se(mol)


def ATSC8se(mol):
    """ATSC8se (BrotoMoreau; ATSC lag=8 prop=se)."""
    return _ATSC8se(mol)


def ATSC0pe(mol):
    """ATSC0pe (BrotoMoreau; ATSC lag=0 prop=pe)."""
    return _ATSC0pe(mol)


def ATSC1pe(mol):
    """ATSC1pe (BrotoMoreau; ATSC lag=1 prop=pe)."""
    return _ATSC1pe(mol)


def ATSC2pe(mol):
    """ATSC2pe (BrotoMoreau; ATSC lag=2 prop=pe)."""
    return _ATSC2pe(mol)


def ATSC3pe(mol):
    """ATSC3pe (BrotoMoreau; ATSC lag=3 prop=pe)."""
    return _ATSC3pe(mol)


def ATSC4pe(mol):
    """ATSC4pe (BrotoMoreau; ATSC lag=4 prop=pe)."""
    return _ATSC4pe(mol)


def ATSC5pe(mol):
    """ATSC5pe (BrotoMoreau; ATSC lag=5 prop=pe)."""
    return _ATSC5pe(mol)


def ATSC6pe(mol):
    """ATSC6pe (BrotoMoreau; ATSC lag=6 prop=pe)."""
    return _ATSC6pe(mol)


def ATSC7pe(mol):
    """ATSC7pe (BrotoMoreau; ATSC lag=7 prop=pe)."""
    return _ATSC7pe(mol)


def ATSC8pe(mol):
    """ATSC8pe (BrotoMoreau; ATSC lag=8 prop=pe)."""
    return _ATSC8pe(mol)


def ATSC0are(mol):
    """ATSC0are (BrotoMoreau; ATSC lag=0 prop=are)."""
    return _ATSC0are(mol)


def ATSC1are(mol):
    """ATSC1are (BrotoMoreau; ATSC lag=1 prop=are)."""
    return _ATSC1are(mol)


def ATSC2are(mol):
    """ATSC2are (BrotoMoreau; ATSC lag=2 prop=are)."""
    return _ATSC2are(mol)


def ATSC3are(mol):
    """ATSC3are (BrotoMoreau; ATSC lag=3 prop=are)."""
    return _ATSC3are(mol)


def ATSC4are(mol):
    """ATSC4are (BrotoMoreau; ATSC lag=4 prop=are)."""
    return _ATSC4are(mol)


def ATSC5are(mol):
    """ATSC5are (BrotoMoreau; ATSC lag=5 prop=are)."""
    return _ATSC5are(mol)


def ATSC6are(mol):
    """ATSC6are (BrotoMoreau; ATSC lag=6 prop=are)."""
    return _ATSC6are(mol)


def ATSC7are(mol):
    """ATSC7are (BrotoMoreau; ATSC lag=7 prop=are)."""
    return _ATSC7are(mol)


def ATSC8are(mol):
    """ATSC8are (BrotoMoreau; ATSC lag=8 prop=are)."""
    return _ATSC8are(mol)


def ATSC0p(mol):
    """ATSC0p (BrotoMoreau; ATSC lag=0 prop=p)."""
    return _ATSC0p(mol)


def ATSC1p(mol):
    """ATSC1p (BrotoMoreau; ATSC lag=1 prop=p)."""
    return _ATSC1p(mol)


def ATSC2p(mol):
    """ATSC2p (BrotoMoreau; ATSC lag=2 prop=p)."""
    return _ATSC2p(mol)


def ATSC3p(mol):
    """ATSC3p (BrotoMoreau; ATSC lag=3 prop=p)."""
    return _ATSC3p(mol)


def ATSC4p(mol):
    """ATSC4p (BrotoMoreau; ATSC lag=4 prop=p)."""
    return _ATSC4p(mol)


def ATSC5p(mol):
    """ATSC5p (BrotoMoreau; ATSC lag=5 prop=p)."""
    return _ATSC5p(mol)


def ATSC6p(mol):
    """ATSC6p (BrotoMoreau; ATSC lag=6 prop=p)."""
    return _ATSC6p(mol)


def ATSC7p(mol):
    """ATSC7p (BrotoMoreau; ATSC lag=7 prop=p)."""
    return _ATSC7p(mol)


def ATSC8p(mol):
    """ATSC8p (BrotoMoreau; ATSC lag=8 prop=p)."""
    return _ATSC8p(mol)


def ATSC0i(mol):
    """ATSC0i (BrotoMoreau; ATSC lag=0 prop=i)."""
    return _ATSC0i(mol)


def ATSC1i(mol):
    """ATSC1i (BrotoMoreau; ATSC lag=1 prop=i)."""
    return _ATSC1i(mol)


def ATSC2i(mol):
    """ATSC2i (BrotoMoreau; ATSC lag=2 prop=i)."""
    return _ATSC2i(mol)


def ATSC3i(mol):
    """ATSC3i (BrotoMoreau; ATSC lag=3 prop=i)."""
    return _ATSC3i(mol)


def ATSC4i(mol):
    """ATSC4i (BrotoMoreau; ATSC lag=4 prop=i)."""
    return _ATSC4i(mol)


def ATSC5i(mol):
    """ATSC5i (BrotoMoreau; ATSC lag=5 prop=i)."""
    return _ATSC5i(mol)


def ATSC6i(mol):
    """ATSC6i (BrotoMoreau; ATSC lag=6 prop=i)."""
    return _ATSC6i(mol)


def ATSC7i(mol):
    """ATSC7i (BrotoMoreau; ATSC lag=7 prop=i)."""
    return _ATSC7i(mol)


def ATSC8i(mol):
    """ATSC8i (BrotoMoreau; ATSC lag=8 prop=i)."""
    return _ATSC8i(mol)


def AATSC0c(mol):
    """AATSC0c (BrotoMoreau; AATSC lag=0 prop=c)."""
    return _AATSC0c(mol)


def AATSC1c(mol):
    """AATSC1c (BrotoMoreau; AATSC lag=1 prop=c)."""
    return _AATSC1c(mol)


def AATSC2c(mol):
    """AATSC2c (BrotoMoreau; AATSC lag=2 prop=c)."""
    return _AATSC2c(mol)


def AATSC3c(mol):
    """AATSC3c (BrotoMoreau; AATSC lag=3 prop=c)."""
    return _AATSC3c(mol)


def AATSC4c(mol):
    """AATSC4c (BrotoMoreau; AATSC lag=4 prop=c)."""
    return _AATSC4c(mol)


def AATSC5c(mol):
    """AATSC5c (BrotoMoreau; AATSC lag=5 prop=c)."""
    return _AATSC5c(mol)


def AATSC6c(mol):
    """AATSC6c (BrotoMoreau; AATSC lag=6 prop=c)."""
    return _AATSC6c(mol)


def AATSC7c(mol):
    """AATSC7c (BrotoMoreau; AATSC lag=7 prop=c)."""
    return _AATSC7c(mol)


def AATSC8c(mol):
    """AATSC8c (BrotoMoreau; AATSC lag=8 prop=c)."""
    return _AATSC8c(mol)


def AATSC0dv(mol):
    """AATSC0dv (BrotoMoreau; AATSC lag=0 prop=dv)."""
    return _AATSC0dv(mol)


def AATSC1dv(mol):
    """AATSC1dv (BrotoMoreau; AATSC lag=1 prop=dv)."""
    return _AATSC1dv(mol)


def AATSC2dv(mol):
    """AATSC2dv (BrotoMoreau; AATSC lag=2 prop=dv)."""
    return _AATSC2dv(mol)


def AATSC3dv(mol):
    """AATSC3dv (BrotoMoreau; AATSC lag=3 prop=dv)."""
    return _AATSC3dv(mol)


def AATSC4dv(mol):
    """AATSC4dv (BrotoMoreau; AATSC lag=4 prop=dv)."""
    return _AATSC4dv(mol)


def AATSC5dv(mol):
    """AATSC5dv (BrotoMoreau; AATSC lag=5 prop=dv)."""
    return _AATSC5dv(mol)


def AATSC6dv(mol):
    """AATSC6dv (BrotoMoreau; AATSC lag=6 prop=dv)."""
    return _AATSC6dv(mol)


def AATSC7dv(mol):
    """AATSC7dv (BrotoMoreau; AATSC lag=7 prop=dv)."""
    return _AATSC7dv(mol)


def AATSC8dv(mol):
    """AATSC8dv (BrotoMoreau; AATSC lag=8 prop=dv)."""
    return _AATSC8dv(mol)


def AATSC0d(mol):
    """AATSC0d (BrotoMoreau; AATSC lag=0 prop=d)."""
    return _AATSC0d(mol)


def AATSC1d(mol):
    """AATSC1d (BrotoMoreau; AATSC lag=1 prop=d)."""
    return _AATSC1d(mol)


def AATSC2d(mol):
    """AATSC2d (BrotoMoreau; AATSC lag=2 prop=d)."""
    return _AATSC2d(mol)


def AATSC3d(mol):
    """AATSC3d (BrotoMoreau; AATSC lag=3 prop=d)."""
    return _AATSC3d(mol)


def AATSC4d(mol):
    """AATSC4d (BrotoMoreau; AATSC lag=4 prop=d)."""
    return _AATSC4d(mol)


def AATSC5d(mol):
    """AATSC5d (BrotoMoreau; AATSC lag=5 prop=d)."""
    return _AATSC5d(mol)


def AATSC6d(mol):
    """AATSC6d (BrotoMoreau; AATSC lag=6 prop=d)."""
    return _AATSC6d(mol)


def AATSC7d(mol):
    """AATSC7d (BrotoMoreau; AATSC lag=7 prop=d)."""
    return _AATSC7d(mol)


def AATSC8d(mol):
    """AATSC8d (BrotoMoreau; AATSC lag=8 prop=d)."""
    return _AATSC8d(mol)


def AATSC0s(mol):
    """AATSC0s (BrotoMoreau; AATSC lag=0 prop=s)."""
    return _AATSC0s(mol)


def AATSC1s(mol):
    """AATSC1s (BrotoMoreau; AATSC lag=1 prop=s)."""
    return _AATSC1s(mol)


def AATSC2s(mol):
    """AATSC2s (BrotoMoreau; AATSC lag=2 prop=s)."""
    return _AATSC2s(mol)


def AATSC3s(mol):
    """AATSC3s (BrotoMoreau; AATSC lag=3 prop=s)."""
    return _AATSC3s(mol)


def AATSC4s(mol):
    """AATSC4s (BrotoMoreau; AATSC lag=4 prop=s)."""
    return _AATSC4s(mol)


def AATSC5s(mol):
    """AATSC5s (BrotoMoreau; AATSC lag=5 prop=s)."""
    return _AATSC5s(mol)


def AATSC6s(mol):
    """AATSC6s (BrotoMoreau; AATSC lag=6 prop=s)."""
    return _AATSC6s(mol)


def AATSC7s(mol):
    """AATSC7s (BrotoMoreau; AATSC lag=7 prop=s)."""
    return _AATSC7s(mol)


def AATSC8s(mol):
    """AATSC8s (BrotoMoreau; AATSC lag=8 prop=s)."""
    return _AATSC8s(mol)


def AATSC0Z(mol):
    """AATSC0Z (BrotoMoreau; AATSC lag=0 prop=Z)."""
    return _AATSC0Z(mol)


def AATSC1Z(mol):
    """AATSC1Z (BrotoMoreau; AATSC lag=1 prop=Z)."""
    return _AATSC1Z(mol)


def AATSC2Z(mol):
    """AATSC2Z (BrotoMoreau; AATSC lag=2 prop=Z)."""
    return _AATSC2Z(mol)


def AATSC3Z(mol):
    """AATSC3Z (BrotoMoreau; AATSC lag=3 prop=Z)."""
    return _AATSC3Z(mol)


def AATSC4Z(mol):
    """AATSC4Z (BrotoMoreau; AATSC lag=4 prop=Z)."""
    return _AATSC4Z(mol)


def AATSC5Z(mol):
    """AATSC5Z (BrotoMoreau; AATSC lag=5 prop=Z)."""
    return _AATSC5Z(mol)


def AATSC6Z(mol):
    """AATSC6Z (BrotoMoreau; AATSC lag=6 prop=Z)."""
    return _AATSC6Z(mol)


def AATSC7Z(mol):
    """AATSC7Z (BrotoMoreau; AATSC lag=7 prop=Z)."""
    return _AATSC7Z(mol)


def AATSC8Z(mol):
    """AATSC8Z (BrotoMoreau; AATSC lag=8 prop=Z)."""
    return _AATSC8Z(mol)


def AATSC0m(mol):
    """AATSC0m (BrotoMoreau; AATSC lag=0 prop=m)."""
    return _AATSC0m(mol)


def AATSC1m(mol):
    """AATSC1m (BrotoMoreau; AATSC lag=1 prop=m)."""
    return _AATSC1m(mol)


def AATSC2m(mol):
    """AATSC2m (BrotoMoreau; AATSC lag=2 prop=m)."""
    return _AATSC2m(mol)


def AATSC3m(mol):
    """AATSC3m (BrotoMoreau; AATSC lag=3 prop=m)."""
    return _AATSC3m(mol)


def AATSC4m(mol):
    """AATSC4m (BrotoMoreau; AATSC lag=4 prop=m)."""
    return _AATSC4m(mol)


def AATSC5m(mol):
    """AATSC5m (BrotoMoreau; AATSC lag=5 prop=m)."""
    return _AATSC5m(mol)


def AATSC6m(mol):
    """AATSC6m (BrotoMoreau; AATSC lag=6 prop=m)."""
    return _AATSC6m(mol)


def AATSC7m(mol):
    """AATSC7m (BrotoMoreau; AATSC lag=7 prop=m)."""
    return _AATSC7m(mol)


def AATSC8m(mol):
    """AATSC8m (BrotoMoreau; AATSC lag=8 prop=m)."""
    return _AATSC8m(mol)


def AATSC0v(mol):
    """AATSC0v (BrotoMoreau; AATSC lag=0 prop=v)."""
    return _AATSC0v(mol)


def AATSC1v(mol):
    """AATSC1v (BrotoMoreau; AATSC lag=1 prop=v)."""
    return _AATSC1v(mol)


def AATSC2v(mol):
    """AATSC2v (BrotoMoreau; AATSC lag=2 prop=v)."""
    return _AATSC2v(mol)


def AATSC3v(mol):
    """AATSC3v (BrotoMoreau; AATSC lag=3 prop=v)."""
    return _AATSC3v(mol)


def AATSC4v(mol):
    """AATSC4v (BrotoMoreau; AATSC lag=4 prop=v)."""
    return _AATSC4v(mol)


def AATSC5v(mol):
    """AATSC5v (BrotoMoreau; AATSC lag=5 prop=v)."""
    return _AATSC5v(mol)


def AATSC6v(mol):
    """AATSC6v (BrotoMoreau; AATSC lag=6 prop=v)."""
    return _AATSC6v(mol)


def AATSC7v(mol):
    """AATSC7v (BrotoMoreau; AATSC lag=7 prop=v)."""
    return _AATSC7v(mol)


def AATSC8v(mol):
    """AATSC8v (BrotoMoreau; AATSC lag=8 prop=v)."""
    return _AATSC8v(mol)


def AATSC0se(mol):
    """AATSC0se (BrotoMoreau; AATSC lag=0 prop=se)."""
    return _AATSC0se(mol)


def AATSC1se(mol):
    """AATSC1se (BrotoMoreau; AATSC lag=1 prop=se)."""
    return _AATSC1se(mol)


def AATSC2se(mol):
    """AATSC2se (BrotoMoreau; AATSC lag=2 prop=se)."""
    return _AATSC2se(mol)


def AATSC3se(mol):
    """AATSC3se (BrotoMoreau; AATSC lag=3 prop=se)."""
    return _AATSC3se(mol)


def AATSC4se(mol):
    """AATSC4se (BrotoMoreau; AATSC lag=4 prop=se)."""
    return _AATSC4se(mol)


def AATSC5se(mol):
    """AATSC5se (BrotoMoreau; AATSC lag=5 prop=se)."""
    return _AATSC5se(mol)


def AATSC6se(mol):
    """AATSC6se (BrotoMoreau; AATSC lag=6 prop=se)."""
    return _AATSC6se(mol)


def AATSC7se(mol):
    """AATSC7se (BrotoMoreau; AATSC lag=7 prop=se)."""
    return _AATSC7se(mol)


def AATSC8se(mol):
    """AATSC8se (BrotoMoreau; AATSC lag=8 prop=se)."""
    return _AATSC8se(mol)


def AATSC0pe(mol):
    """AATSC0pe (BrotoMoreau; AATSC lag=0 prop=pe)."""
    return _AATSC0pe(mol)


def AATSC1pe(mol):
    """AATSC1pe (BrotoMoreau; AATSC lag=1 prop=pe)."""
    return _AATSC1pe(mol)


def AATSC2pe(mol):
    """AATSC2pe (BrotoMoreau; AATSC lag=2 prop=pe)."""
    return _AATSC2pe(mol)


def AATSC3pe(mol):
    """AATSC3pe (BrotoMoreau; AATSC lag=3 prop=pe)."""
    return _AATSC3pe(mol)


def AATSC4pe(mol):
    """AATSC4pe (BrotoMoreau; AATSC lag=4 prop=pe)."""
    return _AATSC4pe(mol)


def AATSC5pe(mol):
    """AATSC5pe (BrotoMoreau; AATSC lag=5 prop=pe)."""
    return _AATSC5pe(mol)


def AATSC6pe(mol):
    """AATSC6pe (BrotoMoreau; AATSC lag=6 prop=pe)."""
    return _AATSC6pe(mol)


def AATSC7pe(mol):
    """AATSC7pe (BrotoMoreau; AATSC lag=7 prop=pe)."""
    return _AATSC7pe(mol)


def AATSC8pe(mol):
    """AATSC8pe (BrotoMoreau; AATSC lag=8 prop=pe)."""
    return _AATSC8pe(mol)


def AATSC0are(mol):
    """AATSC0are (BrotoMoreau; AATSC lag=0 prop=are)."""
    return _AATSC0are(mol)


def AATSC1are(mol):
    """AATSC1are (BrotoMoreau; AATSC lag=1 prop=are)."""
    return _AATSC1are(mol)


def AATSC2are(mol):
    """AATSC2are (BrotoMoreau; AATSC lag=2 prop=are)."""
    return _AATSC2are(mol)


def AATSC3are(mol):
    """AATSC3are (BrotoMoreau; AATSC lag=3 prop=are)."""
    return _AATSC3are(mol)


def AATSC4are(mol):
    """AATSC4are (BrotoMoreau; AATSC lag=4 prop=are)."""
    return _AATSC4are(mol)


def AATSC5are(mol):
    """AATSC5are (BrotoMoreau; AATSC lag=5 prop=are)."""
    return _AATSC5are(mol)


def AATSC6are(mol):
    """AATSC6are (BrotoMoreau; AATSC lag=6 prop=are)."""
    return _AATSC6are(mol)


def AATSC7are(mol):
    """AATSC7are (BrotoMoreau; AATSC lag=7 prop=are)."""
    return _AATSC7are(mol)


def AATSC8are(mol):
    """AATSC8are (BrotoMoreau; AATSC lag=8 prop=are)."""
    return _AATSC8are(mol)


def AATSC0p(mol):
    """AATSC0p (BrotoMoreau; AATSC lag=0 prop=p)."""
    return _AATSC0p(mol)


def AATSC1p(mol):
    """AATSC1p (BrotoMoreau; AATSC lag=1 prop=p)."""
    return _AATSC1p(mol)


def AATSC2p(mol):
    """AATSC2p (BrotoMoreau; AATSC lag=2 prop=p)."""
    return _AATSC2p(mol)


def AATSC3p(mol):
    """AATSC3p (BrotoMoreau; AATSC lag=3 prop=p)."""
    return _AATSC3p(mol)


def AATSC4p(mol):
    """AATSC4p (BrotoMoreau; AATSC lag=4 prop=p)."""
    return _AATSC4p(mol)


def AATSC5p(mol):
    """AATSC5p (BrotoMoreau; AATSC lag=5 prop=p)."""
    return _AATSC5p(mol)


def AATSC6p(mol):
    """AATSC6p (BrotoMoreau; AATSC lag=6 prop=p)."""
    return _AATSC6p(mol)


def AATSC7p(mol):
    """AATSC7p (BrotoMoreau; AATSC lag=7 prop=p)."""
    return _AATSC7p(mol)


def AATSC8p(mol):
    """AATSC8p (BrotoMoreau; AATSC lag=8 prop=p)."""
    return _AATSC8p(mol)


def AATSC0i(mol):
    """AATSC0i (BrotoMoreau; AATSC lag=0 prop=i)."""
    return _AATSC0i(mol)


def AATSC1i(mol):
    """AATSC1i (BrotoMoreau; AATSC lag=1 prop=i)."""
    return _AATSC1i(mol)


def AATSC2i(mol):
    """AATSC2i (BrotoMoreau; AATSC lag=2 prop=i)."""
    return _AATSC2i(mol)


def AATSC3i(mol):
    """AATSC3i (BrotoMoreau; AATSC lag=3 prop=i)."""
    return _AATSC3i(mol)


def AATSC4i(mol):
    """AATSC4i (BrotoMoreau; AATSC lag=4 prop=i)."""
    return _AATSC4i(mol)


def AATSC5i(mol):
    """AATSC5i (BrotoMoreau; AATSC lag=5 prop=i)."""
    return _AATSC5i(mol)


def AATSC6i(mol):
    """AATSC6i (BrotoMoreau; AATSC lag=6 prop=i)."""
    return _AATSC6i(mol)


def AATSC7i(mol):
    """AATSC7i (BrotoMoreau; AATSC lag=7 prop=i)."""
    return _AATSC7i(mol)


def AATSC8i(mol):
    """AATSC8i (BrotoMoreau; AATSC lag=8 prop=i)."""
    return _AATSC8i(mol)


def MATS1c(mol):
    """MATS1c (MoranGeary; MATS lag=1 prop=c)."""
    return _MATS1c(mol)


def MATS2c(mol):
    """MATS2c (MoranGeary; MATS lag=2 prop=c)."""
    return _MATS2c(mol)


def MATS3c(mol):
    """MATS3c (MoranGeary; MATS lag=3 prop=c)."""
    return _MATS3c(mol)


def MATS4c(mol):
    """MATS4c (MoranGeary; MATS lag=4 prop=c)."""
    return _MATS4c(mol)


def MATS5c(mol):
    """MATS5c (MoranGeary; MATS lag=5 prop=c)."""
    return _MATS5c(mol)


def MATS6c(mol):
    """MATS6c (MoranGeary; MATS lag=6 prop=c)."""
    return _MATS6c(mol)


def MATS7c(mol):
    """MATS7c (MoranGeary; MATS lag=7 prop=c)."""
    return _MATS7c(mol)


def MATS8c(mol):
    """MATS8c (MoranGeary; MATS lag=8 prop=c)."""
    return _MATS8c(mol)


def MATS1dv(mol):
    """MATS1dv (MoranGeary; MATS lag=1 prop=dv)."""
    return _MATS1dv(mol)


def MATS2dv(mol):
    """MATS2dv (MoranGeary; MATS lag=2 prop=dv)."""
    return _MATS2dv(mol)


def MATS3dv(mol):
    """MATS3dv (MoranGeary; MATS lag=3 prop=dv)."""
    return _MATS3dv(mol)


def MATS4dv(mol):
    """MATS4dv (MoranGeary; MATS lag=4 prop=dv)."""
    return _MATS4dv(mol)


def MATS5dv(mol):
    """MATS5dv (MoranGeary; MATS lag=5 prop=dv)."""
    return _MATS5dv(mol)


def MATS6dv(mol):
    """MATS6dv (MoranGeary; MATS lag=6 prop=dv)."""
    return _MATS6dv(mol)


def MATS7dv(mol):
    """MATS7dv (MoranGeary; MATS lag=7 prop=dv)."""
    return _MATS7dv(mol)


def MATS8dv(mol):
    """MATS8dv (MoranGeary; MATS lag=8 prop=dv)."""
    return _MATS8dv(mol)


def MATS1d(mol):
    """MATS1d (MoranGeary; MATS lag=1 prop=d)."""
    return _MATS1d(mol)


def MATS2d(mol):
    """MATS2d (MoranGeary; MATS lag=2 prop=d)."""
    return _MATS2d(mol)


def MATS3d(mol):
    """MATS3d (MoranGeary; MATS lag=3 prop=d)."""
    return _MATS3d(mol)


def MATS4d(mol):
    """MATS4d (MoranGeary; MATS lag=4 prop=d)."""
    return _MATS4d(mol)


def MATS5d(mol):
    """MATS5d (MoranGeary; MATS lag=5 prop=d)."""
    return _MATS5d(mol)


def MATS6d(mol):
    """MATS6d (MoranGeary; MATS lag=6 prop=d)."""
    return _MATS6d(mol)


def MATS7d(mol):
    """MATS7d (MoranGeary; MATS lag=7 prop=d)."""
    return _MATS7d(mol)


def MATS8d(mol):
    """MATS8d (MoranGeary; MATS lag=8 prop=d)."""
    return _MATS8d(mol)


def MATS1s(mol):
    """MATS1s (MoranGeary; MATS lag=1 prop=s)."""
    return _MATS1s(mol)


def MATS2s(mol):
    """MATS2s (MoranGeary; MATS lag=2 prop=s)."""
    return _MATS2s(mol)


def MATS3s(mol):
    """MATS3s (MoranGeary; MATS lag=3 prop=s)."""
    return _MATS3s(mol)


def MATS4s(mol):
    """MATS4s (MoranGeary; MATS lag=4 prop=s)."""
    return _MATS4s(mol)


def MATS5s(mol):
    """MATS5s (MoranGeary; MATS lag=5 prop=s)."""
    return _MATS5s(mol)


def MATS6s(mol):
    """MATS6s (MoranGeary; MATS lag=6 prop=s)."""
    return _MATS6s(mol)


def MATS7s(mol):
    """MATS7s (MoranGeary; MATS lag=7 prop=s)."""
    return _MATS7s(mol)


def MATS8s(mol):
    """MATS8s (MoranGeary; MATS lag=8 prop=s)."""
    return _MATS8s(mol)


def MATS1Z(mol):
    """MATS1Z (MoranGeary; MATS lag=1 prop=Z)."""
    return _MATS1Z(mol)


def MATS2Z(mol):
    """MATS2Z (MoranGeary; MATS lag=2 prop=Z)."""
    return _MATS2Z(mol)


def MATS3Z(mol):
    """MATS3Z (MoranGeary; MATS lag=3 prop=Z)."""
    return _MATS3Z(mol)


def MATS4Z(mol):
    """MATS4Z (MoranGeary; MATS lag=4 prop=Z)."""
    return _MATS4Z(mol)


def MATS5Z(mol):
    """MATS5Z (MoranGeary; MATS lag=5 prop=Z)."""
    return _MATS5Z(mol)


def MATS6Z(mol):
    """MATS6Z (MoranGeary; MATS lag=6 prop=Z)."""
    return _MATS6Z(mol)


def MATS7Z(mol):
    """MATS7Z (MoranGeary; MATS lag=7 prop=Z)."""
    return _MATS7Z(mol)


def MATS8Z(mol):
    """MATS8Z (MoranGeary; MATS lag=8 prop=Z)."""
    return _MATS8Z(mol)


def MATS1m(mol):
    """MATS1m (MoranGeary; MATS lag=1 prop=m)."""
    return _MATS1m(mol)


def MATS2m(mol):
    """MATS2m (MoranGeary; MATS lag=2 prop=m)."""
    return _MATS2m(mol)


def MATS3m(mol):
    """MATS3m (MoranGeary; MATS lag=3 prop=m)."""
    return _MATS3m(mol)


def MATS4m(mol):
    """MATS4m (MoranGeary; MATS lag=4 prop=m)."""
    return _MATS4m(mol)


def MATS5m(mol):
    """MATS5m (MoranGeary; MATS lag=5 prop=m)."""
    return _MATS5m(mol)


def MATS6m(mol):
    """MATS6m (MoranGeary; MATS lag=6 prop=m)."""
    return _MATS6m(mol)


def MATS7m(mol):
    """MATS7m (MoranGeary; MATS lag=7 prop=m)."""
    return _MATS7m(mol)


def MATS8m(mol):
    """MATS8m (MoranGeary; MATS lag=8 prop=m)."""
    return _MATS8m(mol)


def MATS1v(mol):
    """MATS1v (MoranGeary; MATS lag=1 prop=v)."""
    return _MATS1v(mol)


def MATS2v(mol):
    """MATS2v (MoranGeary; MATS lag=2 prop=v)."""
    return _MATS2v(mol)


def MATS3v(mol):
    """MATS3v (MoranGeary; MATS lag=3 prop=v)."""
    return _MATS3v(mol)


def MATS4v(mol):
    """MATS4v (MoranGeary; MATS lag=4 prop=v)."""
    return _MATS4v(mol)


def MATS5v(mol):
    """MATS5v (MoranGeary; MATS lag=5 prop=v)."""
    return _MATS5v(mol)


def MATS6v(mol):
    """MATS6v (MoranGeary; MATS lag=6 prop=v)."""
    return _MATS6v(mol)


def MATS7v(mol):
    """MATS7v (MoranGeary; MATS lag=7 prop=v)."""
    return _MATS7v(mol)


def MATS8v(mol):
    """MATS8v (MoranGeary; MATS lag=8 prop=v)."""
    return _MATS8v(mol)


def MATS1se(mol):
    """MATS1se (MoranGeary; MATS lag=1 prop=se)."""
    return _MATS1se(mol)


def MATS2se(mol):
    """MATS2se (MoranGeary; MATS lag=2 prop=se)."""
    return _MATS2se(mol)


def MATS3se(mol):
    """MATS3se (MoranGeary; MATS lag=3 prop=se)."""
    return _MATS3se(mol)


def MATS4se(mol):
    """MATS4se (MoranGeary; MATS lag=4 prop=se)."""
    return _MATS4se(mol)


def MATS5se(mol):
    """MATS5se (MoranGeary; MATS lag=5 prop=se)."""
    return _MATS5se(mol)


def MATS6se(mol):
    """MATS6se (MoranGeary; MATS lag=6 prop=se)."""
    return _MATS6se(mol)


def MATS7se(mol):
    """MATS7se (MoranGeary; MATS lag=7 prop=se)."""
    return _MATS7se(mol)


def MATS8se(mol):
    """MATS8se (MoranGeary; MATS lag=8 prop=se)."""
    return _MATS8se(mol)


def MATS1pe(mol):
    """MATS1pe (MoranGeary; MATS lag=1 prop=pe)."""
    return _MATS1pe(mol)


def MATS2pe(mol):
    """MATS2pe (MoranGeary; MATS lag=2 prop=pe)."""
    return _MATS2pe(mol)


def MATS3pe(mol):
    """MATS3pe (MoranGeary; MATS lag=3 prop=pe)."""
    return _MATS3pe(mol)


def MATS4pe(mol):
    """MATS4pe (MoranGeary; MATS lag=4 prop=pe)."""
    return _MATS4pe(mol)


def MATS5pe(mol):
    """MATS5pe (MoranGeary; MATS lag=5 prop=pe)."""
    return _MATS5pe(mol)


def MATS6pe(mol):
    """MATS6pe (MoranGeary; MATS lag=6 prop=pe)."""
    return _MATS6pe(mol)


def MATS7pe(mol):
    """MATS7pe (MoranGeary; MATS lag=7 prop=pe)."""
    return _MATS7pe(mol)


def MATS8pe(mol):
    """MATS8pe (MoranGeary; MATS lag=8 prop=pe)."""
    return _MATS8pe(mol)


def MATS1are(mol):
    """MATS1are (MoranGeary; MATS lag=1 prop=are)."""
    return _MATS1are(mol)


def MATS2are(mol):
    """MATS2are (MoranGeary; MATS lag=2 prop=are)."""
    return _MATS2are(mol)


def MATS3are(mol):
    """MATS3are (MoranGeary; MATS lag=3 prop=are)."""
    return _MATS3are(mol)


def MATS4are(mol):
    """MATS4are (MoranGeary; MATS lag=4 prop=are)."""
    return _MATS4are(mol)


def MATS5are(mol):
    """MATS5are (MoranGeary; MATS lag=5 prop=are)."""
    return _MATS5are(mol)


def MATS6are(mol):
    """MATS6are (MoranGeary; MATS lag=6 prop=are)."""
    return _MATS6are(mol)


def MATS7are(mol):
    """MATS7are (MoranGeary; MATS lag=7 prop=are)."""
    return _MATS7are(mol)


def MATS8are(mol):
    """MATS8are (MoranGeary; MATS lag=8 prop=are)."""
    return _MATS8are(mol)


def MATS1p(mol):
    """MATS1p (MoranGeary; MATS lag=1 prop=p)."""
    return _MATS1p(mol)


def MATS2p(mol):
    """MATS2p (MoranGeary; MATS lag=2 prop=p)."""
    return _MATS2p(mol)


def MATS3p(mol):
    """MATS3p (MoranGeary; MATS lag=3 prop=p)."""
    return _MATS3p(mol)


def MATS4p(mol):
    """MATS4p (MoranGeary; MATS lag=4 prop=p)."""
    return _MATS4p(mol)


def MATS5p(mol):
    """MATS5p (MoranGeary; MATS lag=5 prop=p)."""
    return _MATS5p(mol)


def MATS6p(mol):
    """MATS6p (MoranGeary; MATS lag=6 prop=p)."""
    return _MATS6p(mol)


def MATS7p(mol):
    """MATS7p (MoranGeary; MATS lag=7 prop=p)."""
    return _MATS7p(mol)


def MATS8p(mol):
    """MATS8p (MoranGeary; MATS lag=8 prop=p)."""
    return _MATS8p(mol)


def MATS1i(mol):
    """MATS1i (MoranGeary; MATS lag=1 prop=i)."""
    return _MATS1i(mol)


def MATS2i(mol):
    """MATS2i (MoranGeary; MATS lag=2 prop=i)."""
    return _MATS2i(mol)


def MATS3i(mol):
    """MATS3i (MoranGeary; MATS lag=3 prop=i)."""
    return _MATS3i(mol)


def MATS4i(mol):
    """MATS4i (MoranGeary; MATS lag=4 prop=i)."""
    return _MATS4i(mol)


def MATS5i(mol):
    """MATS5i (MoranGeary; MATS lag=5 prop=i)."""
    return _MATS5i(mol)


def MATS6i(mol):
    """MATS6i (MoranGeary; MATS lag=6 prop=i)."""
    return _MATS6i(mol)


def MATS7i(mol):
    """MATS7i (MoranGeary; MATS lag=7 prop=i)."""
    return _MATS7i(mol)


def MATS8i(mol):
    """MATS8i (MoranGeary; MATS lag=8 prop=i)."""
    return _MATS8i(mol)


def GATS1c(mol):
    """GATS1c (MoranGeary; GATS lag=1 prop=c)."""
    return _GATS1c(mol)


def GATS2c(mol):
    """GATS2c (MoranGeary; GATS lag=2 prop=c)."""
    return _GATS2c(mol)


def GATS3c(mol):
    """GATS3c (MoranGeary; GATS lag=3 prop=c)."""
    return _GATS3c(mol)


def GATS4c(mol):
    """GATS4c (MoranGeary; GATS lag=4 prop=c)."""
    return _GATS4c(mol)


def GATS5c(mol):
    """GATS5c (MoranGeary; GATS lag=5 prop=c)."""
    return _GATS5c(mol)


def GATS6c(mol):
    """GATS6c (MoranGeary; GATS lag=6 prop=c)."""
    return _GATS6c(mol)


def GATS7c(mol):
    """GATS7c (MoranGeary; GATS lag=7 prop=c)."""
    return _GATS7c(mol)


def GATS8c(mol):
    """GATS8c (MoranGeary; GATS lag=8 prop=c)."""
    return _GATS8c(mol)


def GATS1dv(mol):
    """GATS1dv (MoranGeary; GATS lag=1 prop=dv)."""
    return _GATS1dv(mol)


def GATS2dv(mol):
    """GATS2dv (MoranGeary; GATS lag=2 prop=dv)."""
    return _GATS2dv(mol)


def GATS3dv(mol):
    """GATS3dv (MoranGeary; GATS lag=3 prop=dv)."""
    return _GATS3dv(mol)


def GATS4dv(mol):
    """GATS4dv (MoranGeary; GATS lag=4 prop=dv)."""
    return _GATS4dv(mol)


def GATS5dv(mol):
    """GATS5dv (MoranGeary; GATS lag=5 prop=dv)."""
    return _GATS5dv(mol)


def GATS6dv(mol):
    """GATS6dv (MoranGeary; GATS lag=6 prop=dv)."""
    return _GATS6dv(mol)


def GATS7dv(mol):
    """GATS7dv (MoranGeary; GATS lag=7 prop=dv)."""
    return _GATS7dv(mol)


def GATS8dv(mol):
    """GATS8dv (MoranGeary; GATS lag=8 prop=dv)."""
    return _GATS8dv(mol)


def GATS1d(mol):
    """GATS1d (MoranGeary; GATS lag=1 prop=d)."""
    return _GATS1d(mol)


def GATS2d(mol):
    """GATS2d (MoranGeary; GATS lag=2 prop=d)."""
    return _GATS2d(mol)


def GATS3d(mol):
    """GATS3d (MoranGeary; GATS lag=3 prop=d)."""
    return _GATS3d(mol)


def GATS4d(mol):
    """GATS4d (MoranGeary; GATS lag=4 prop=d)."""
    return _GATS4d(mol)


def GATS5d(mol):
    """GATS5d (MoranGeary; GATS lag=5 prop=d)."""
    return _GATS5d(mol)


def GATS6d(mol):
    """GATS6d (MoranGeary; GATS lag=6 prop=d)."""
    return _GATS6d(mol)


def GATS7d(mol):
    """GATS7d (MoranGeary; GATS lag=7 prop=d)."""
    return _GATS7d(mol)


def GATS8d(mol):
    """GATS8d (MoranGeary; GATS lag=8 prop=d)."""
    return _GATS8d(mol)


def GATS1s(mol):
    """GATS1s (MoranGeary; GATS lag=1 prop=s)."""
    return _GATS1s(mol)


def GATS2s(mol):
    """GATS2s (MoranGeary; GATS lag=2 prop=s)."""
    return _GATS2s(mol)


def GATS3s(mol):
    """GATS3s (MoranGeary; GATS lag=3 prop=s)."""
    return _GATS3s(mol)


def GATS4s(mol):
    """GATS4s (MoranGeary; GATS lag=4 prop=s)."""
    return _GATS4s(mol)


def GATS5s(mol):
    """GATS5s (MoranGeary; GATS lag=5 prop=s)."""
    return _GATS5s(mol)


def GATS6s(mol):
    """GATS6s (MoranGeary; GATS lag=6 prop=s)."""
    return _GATS6s(mol)


def GATS7s(mol):
    """GATS7s (MoranGeary; GATS lag=7 prop=s)."""
    return _GATS7s(mol)


def GATS8s(mol):
    """GATS8s (MoranGeary; GATS lag=8 prop=s)."""
    return _GATS8s(mol)


def GATS1Z(mol):
    """GATS1Z (MoranGeary; GATS lag=1 prop=Z)."""
    return _GATS1Z(mol)


def GATS2Z(mol):
    """GATS2Z (MoranGeary; GATS lag=2 prop=Z)."""
    return _GATS2Z(mol)


def GATS3Z(mol):
    """GATS3Z (MoranGeary; GATS lag=3 prop=Z)."""
    return _GATS3Z(mol)


def GATS4Z(mol):
    """GATS4Z (MoranGeary; GATS lag=4 prop=Z)."""
    return _GATS4Z(mol)


def GATS5Z(mol):
    """GATS5Z (MoranGeary; GATS lag=5 prop=Z)."""
    return _GATS5Z(mol)


def GATS6Z(mol):
    """GATS6Z (MoranGeary; GATS lag=6 prop=Z)."""
    return _GATS6Z(mol)


def GATS7Z(mol):
    """GATS7Z (MoranGeary; GATS lag=7 prop=Z)."""
    return _GATS7Z(mol)


def GATS8Z(mol):
    """GATS8Z (MoranGeary; GATS lag=8 prop=Z)."""
    return _GATS8Z(mol)


def GATS1m(mol):
    """GATS1m (MoranGeary; GATS lag=1 prop=m)."""
    return _GATS1m(mol)


def GATS2m(mol):
    """GATS2m (MoranGeary; GATS lag=2 prop=m)."""
    return _GATS2m(mol)


def GATS3m(mol):
    """GATS3m (MoranGeary; GATS lag=3 prop=m)."""
    return _GATS3m(mol)


def GATS4m(mol):
    """GATS4m (MoranGeary; GATS lag=4 prop=m)."""
    return _GATS4m(mol)


def GATS5m(mol):
    """GATS5m (MoranGeary; GATS lag=5 prop=m)."""
    return _GATS5m(mol)


def GATS6m(mol):
    """GATS6m (MoranGeary; GATS lag=6 prop=m)."""
    return _GATS6m(mol)


def GATS7m(mol):
    """GATS7m (MoranGeary; GATS lag=7 prop=m)."""
    return _GATS7m(mol)


def GATS8m(mol):
    """GATS8m (MoranGeary; GATS lag=8 prop=m)."""
    return _GATS8m(mol)


def GATS1v(mol):
    """GATS1v (MoranGeary; GATS lag=1 prop=v)."""
    return _GATS1v(mol)


def GATS2v(mol):
    """GATS2v (MoranGeary; GATS lag=2 prop=v)."""
    return _GATS2v(mol)


def GATS3v(mol):
    """GATS3v (MoranGeary; GATS lag=3 prop=v)."""
    return _GATS3v(mol)


def GATS4v(mol):
    """GATS4v (MoranGeary; GATS lag=4 prop=v)."""
    return _GATS4v(mol)


def GATS5v(mol):
    """GATS5v (MoranGeary; GATS lag=5 prop=v)."""
    return _GATS5v(mol)


def GATS6v(mol):
    """GATS6v (MoranGeary; GATS lag=6 prop=v)."""
    return _GATS6v(mol)


def GATS7v(mol):
    """GATS7v (MoranGeary; GATS lag=7 prop=v)."""
    return _GATS7v(mol)


def GATS8v(mol):
    """GATS8v (MoranGeary; GATS lag=8 prop=v)."""
    return _GATS8v(mol)


def GATS1se(mol):
    """GATS1se (MoranGeary; GATS lag=1 prop=se)."""
    return _GATS1se(mol)


def GATS2se(mol):
    """GATS2se (MoranGeary; GATS lag=2 prop=se)."""
    return _GATS2se(mol)


def GATS3se(mol):
    """GATS3se (MoranGeary; GATS lag=3 prop=se)."""
    return _GATS3se(mol)


def GATS4se(mol):
    """GATS4se (MoranGeary; GATS lag=4 prop=se)."""
    return _GATS4se(mol)


def GATS5se(mol):
    """GATS5se (MoranGeary; GATS lag=5 prop=se)."""
    return _GATS5se(mol)


def GATS6se(mol):
    """GATS6se (MoranGeary; GATS lag=6 prop=se)."""
    return _GATS6se(mol)


def GATS7se(mol):
    """GATS7se (MoranGeary; GATS lag=7 prop=se)."""
    return _GATS7se(mol)


def GATS8se(mol):
    """GATS8se (MoranGeary; GATS lag=8 prop=se)."""
    return _GATS8se(mol)


def GATS1pe(mol):
    """GATS1pe (MoranGeary; GATS lag=1 prop=pe)."""
    return _GATS1pe(mol)


def GATS2pe(mol):
    """GATS2pe (MoranGeary; GATS lag=2 prop=pe)."""
    return _GATS2pe(mol)


def GATS3pe(mol):
    """GATS3pe (MoranGeary; GATS lag=3 prop=pe)."""
    return _GATS3pe(mol)


def GATS4pe(mol):
    """GATS4pe (MoranGeary; GATS lag=4 prop=pe)."""
    return _GATS4pe(mol)


def GATS5pe(mol):
    """GATS5pe (MoranGeary; GATS lag=5 prop=pe)."""
    return _GATS5pe(mol)


def GATS6pe(mol):
    """GATS6pe (MoranGeary; GATS lag=6 prop=pe)."""
    return _GATS6pe(mol)


def GATS7pe(mol):
    """GATS7pe (MoranGeary; GATS lag=7 prop=pe)."""
    return _GATS7pe(mol)


def GATS8pe(mol):
    """GATS8pe (MoranGeary; GATS lag=8 prop=pe)."""
    return _GATS8pe(mol)


def GATS1are(mol):
    """GATS1are (MoranGeary; GATS lag=1 prop=are)."""
    return _GATS1are(mol)


def GATS2are(mol):
    """GATS2are (MoranGeary; GATS lag=2 prop=are)."""
    return _GATS2are(mol)


def GATS3are(mol):
    """GATS3are (MoranGeary; GATS lag=3 prop=are)."""
    return _GATS3are(mol)


def GATS4are(mol):
    """GATS4are (MoranGeary; GATS lag=4 prop=are)."""
    return _GATS4are(mol)


def GATS5are(mol):
    """GATS5are (MoranGeary; GATS lag=5 prop=are)."""
    return _GATS5are(mol)


def GATS6are(mol):
    """GATS6are (MoranGeary; GATS lag=6 prop=are)."""
    return _GATS6are(mol)


def GATS7are(mol):
    """GATS7are (MoranGeary; GATS lag=7 prop=are)."""
    return _GATS7are(mol)


def GATS8are(mol):
    """GATS8are (MoranGeary; GATS lag=8 prop=are)."""
    return _GATS8are(mol)


def GATS1p(mol):
    """GATS1p (MoranGeary; GATS lag=1 prop=p)."""
    return _GATS1p(mol)


def GATS2p(mol):
    """GATS2p (MoranGeary; GATS lag=2 prop=p)."""
    return _GATS2p(mol)


def GATS3p(mol):
    """GATS3p (MoranGeary; GATS lag=3 prop=p)."""
    return _GATS3p(mol)


def GATS4p(mol):
    """GATS4p (MoranGeary; GATS lag=4 prop=p)."""
    return _GATS4p(mol)


def GATS5p(mol):
    """GATS5p (MoranGeary; GATS lag=5 prop=p)."""
    return _GATS5p(mol)


def GATS6p(mol):
    """GATS6p (MoranGeary; GATS lag=6 prop=p)."""
    return _GATS6p(mol)


def GATS7p(mol):
    """GATS7p (MoranGeary; GATS lag=7 prop=p)."""
    return _GATS7p(mol)


def GATS8p(mol):
    """GATS8p (MoranGeary; GATS lag=8 prop=p)."""
    return _GATS8p(mol)


def GATS1i(mol):
    """GATS1i (MoranGeary; GATS lag=1 prop=i)."""
    return _GATS1i(mol)


def GATS2i(mol):
    """GATS2i (MoranGeary; GATS lag=2 prop=i)."""
    return _GATS2i(mol)


def GATS3i(mol):
    """GATS3i (MoranGeary; GATS lag=3 prop=i)."""
    return _GATS3i(mol)


def GATS4i(mol):
    """GATS4i (MoranGeary; GATS lag=4 prop=i)."""
    return _GATS4i(mol)


def GATS5i(mol):
    """GATS5i (MoranGeary; GATS lag=5 prop=i)."""
    return _GATS5i(mol)


def GATS6i(mol):
    """GATS6i (MoranGeary; GATS lag=6 prop=i)."""
    return _GATS6i(mol)


def GATS7i(mol):
    """GATS7i (MoranGeary; GATS lag=7 prop=i)."""
    return _GATS7i(mol)


def GATS8i(mol):
    """GATS8i (MoranGeary; GATS lag=8 prop=i)."""
    return _GATS8i(mol)


ALL_DESCRIPTOR_NAMES = (
    "ATS0dv",
    "ATS1dv",
    "ATS2dv",
    "ATS3dv",
    "ATS4dv",
    "ATS5dv",
    "ATS6dv",
    "ATS7dv",
    "ATS8dv",
    "ATS0d",
    "ATS1d",
    "ATS2d",
    "ATS3d",
    "ATS4d",
    "ATS5d",
    "ATS6d",
    "ATS7d",
    "ATS8d",
    "ATS0s",
    "ATS1s",
    "ATS2s",
    "ATS3s",
    "ATS4s",
    "ATS5s",
    "ATS6s",
    "ATS7s",
    "ATS8s",
    "ATS0Z",
    "ATS1Z",
    "ATS2Z",
    "ATS3Z",
    "ATS4Z",
    "ATS5Z",
    "ATS6Z",
    "ATS7Z",
    "ATS8Z",
    "ATS0m",
    "ATS1m",
    "ATS2m",
    "ATS3m",
    "ATS4m",
    "ATS5m",
    "ATS6m",
    "ATS7m",
    "ATS8m",
    "ATS0v",
    "ATS1v",
    "ATS2v",
    "ATS3v",
    "ATS4v",
    "ATS5v",
    "ATS6v",
    "ATS7v",
    "ATS8v",
    "ATS0se",
    "ATS1se",
    "ATS2se",
    "ATS3se",
    "ATS4se",
    "ATS5se",
    "ATS6se",
    "ATS7se",
    "ATS8se",
    "ATS0pe",
    "ATS1pe",
    "ATS2pe",
    "ATS3pe",
    "ATS4pe",
    "ATS5pe",
    "ATS6pe",
    "ATS7pe",
    "ATS8pe",
    "ATS0are",
    "ATS1are",
    "ATS2are",
    "ATS3are",
    "ATS4are",
    "ATS5are",
    "ATS6are",
    "ATS7are",
    "ATS8are",
    "ATS0p",
    "ATS1p",
    "ATS2p",
    "ATS3p",
    "ATS4p",
    "ATS5p",
    "ATS6p",
    "ATS7p",
    "ATS8p",
    "ATS0i",
    "ATS1i",
    "ATS2i",
    "ATS3i",
    "ATS4i",
    "ATS5i",
    "ATS6i",
    "ATS7i",
    "ATS8i",
    "AATS0dv",
    "AATS1dv",
    "AATS2dv",
    "AATS3dv",
    "AATS4dv",
    "AATS5dv",
    "AATS6dv",
    "AATS7dv",
    "AATS8dv",
    "AATS0d",
    "AATS1d",
    "AATS2d",
    "AATS3d",
    "AATS4d",
    "AATS5d",
    "AATS6d",
    "AATS7d",
    "AATS8d",
    "AATS0s",
    "AATS1s",
    "AATS2s",
    "AATS3s",
    "AATS4s",
    "AATS5s",
    "AATS6s",
    "AATS7s",
    "AATS8s",
    "AATS0Z",
    "AATS1Z",
    "AATS2Z",
    "AATS3Z",
    "AATS4Z",
    "AATS5Z",
    "AATS6Z",
    "AATS7Z",
    "AATS8Z",
    "AATS0m",
    "AATS1m",
    "AATS2m",
    "AATS3m",
    "AATS4m",
    "AATS5m",
    "AATS6m",
    "AATS7m",
    "AATS8m",
    "AATS0v",
    "AATS1v",
    "AATS2v",
    "AATS3v",
    "AATS4v",
    "AATS5v",
    "AATS6v",
    "AATS7v",
    "AATS8v",
    "AATS0se",
    "AATS1se",
    "AATS2se",
    "AATS3se",
    "AATS4se",
    "AATS5se",
    "AATS6se",
    "AATS7se",
    "AATS8se",
    "AATS0pe",
    "AATS1pe",
    "AATS2pe",
    "AATS3pe",
    "AATS4pe",
    "AATS5pe",
    "AATS6pe",
    "AATS7pe",
    "AATS8pe",
    "AATS0are",
    "AATS1are",
    "AATS2are",
    "AATS3are",
    "AATS4are",
    "AATS5are",
    "AATS6are",
    "AATS7are",
    "AATS8are",
    "AATS0p",
    "AATS1p",
    "AATS2p",
    "AATS3p",
    "AATS4p",
    "AATS5p",
    "AATS6p",
    "AATS7p",
    "AATS8p",
    "AATS0i",
    "AATS1i",
    "AATS2i",
    "AATS3i",
    "AATS4i",
    "AATS5i",
    "AATS6i",
    "AATS7i",
    "AATS8i",
    "ATSC0c",
    "ATSC1c",
    "ATSC2c",
    "ATSC3c",
    "ATSC4c",
    "ATSC5c",
    "ATSC6c",
    "ATSC7c",
    "ATSC8c",
    "ATSC0dv",
    "ATSC1dv",
    "ATSC2dv",
    "ATSC3dv",
    "ATSC4dv",
    "ATSC5dv",
    "ATSC6dv",
    "ATSC7dv",
    "ATSC8dv",
    "ATSC0d",
    "ATSC1d",
    "ATSC2d",
    "ATSC3d",
    "ATSC4d",
    "ATSC5d",
    "ATSC6d",
    "ATSC7d",
    "ATSC8d",
    "ATSC0s",
    "ATSC1s",
    "ATSC2s",
    "ATSC3s",
    "ATSC4s",
    "ATSC5s",
    "ATSC6s",
    "ATSC7s",
    "ATSC8s",
    "ATSC0Z",
    "ATSC1Z",
    "ATSC2Z",
    "ATSC3Z",
    "ATSC4Z",
    "ATSC5Z",
    "ATSC6Z",
    "ATSC7Z",
    "ATSC8Z",
    "ATSC0m",
    "ATSC1m",
    "ATSC2m",
    "ATSC3m",
    "ATSC4m",
    "ATSC5m",
    "ATSC6m",
    "ATSC7m",
    "ATSC8m",
    "ATSC0v",
    "ATSC1v",
    "ATSC2v",
    "ATSC3v",
    "ATSC4v",
    "ATSC5v",
    "ATSC6v",
    "ATSC7v",
    "ATSC8v",
    "ATSC0se",
    "ATSC1se",
    "ATSC2se",
    "ATSC3se",
    "ATSC4se",
    "ATSC5se",
    "ATSC6se",
    "ATSC7se",
    "ATSC8se",
    "ATSC0pe",
    "ATSC1pe",
    "ATSC2pe",
    "ATSC3pe",
    "ATSC4pe",
    "ATSC5pe",
    "ATSC6pe",
    "ATSC7pe",
    "ATSC8pe",
    "ATSC0are",
    "ATSC1are",
    "ATSC2are",
    "ATSC3are",
    "ATSC4are",
    "ATSC5are",
    "ATSC6are",
    "ATSC7are",
    "ATSC8are",
    "ATSC0p",
    "ATSC1p",
    "ATSC2p",
    "ATSC3p",
    "ATSC4p",
    "ATSC5p",
    "ATSC6p",
    "ATSC7p",
    "ATSC8p",
    "ATSC0i",
    "ATSC1i",
    "ATSC2i",
    "ATSC3i",
    "ATSC4i",
    "ATSC5i",
    "ATSC6i",
    "ATSC7i",
    "ATSC8i",
    "AATSC0c",
    "AATSC1c",
    "AATSC2c",
    "AATSC3c",
    "AATSC4c",
    "AATSC5c",
    "AATSC6c",
    "AATSC7c",
    "AATSC8c",
    "AATSC0dv",
    "AATSC1dv",
    "AATSC2dv",
    "AATSC3dv",
    "AATSC4dv",
    "AATSC5dv",
    "AATSC6dv",
    "AATSC7dv",
    "AATSC8dv",
    "AATSC0d",
    "AATSC1d",
    "AATSC2d",
    "AATSC3d",
    "AATSC4d",
    "AATSC5d",
    "AATSC6d",
    "AATSC7d",
    "AATSC8d",
    "AATSC0s",
    "AATSC1s",
    "AATSC2s",
    "AATSC3s",
    "AATSC4s",
    "AATSC5s",
    "AATSC6s",
    "AATSC7s",
    "AATSC8s",
    "AATSC0Z",
    "AATSC1Z",
    "AATSC2Z",
    "AATSC3Z",
    "AATSC4Z",
    "AATSC5Z",
    "AATSC6Z",
    "AATSC7Z",
    "AATSC8Z",
    "AATSC0m",
    "AATSC1m",
    "AATSC2m",
    "AATSC3m",
    "AATSC4m",
    "AATSC5m",
    "AATSC6m",
    "AATSC7m",
    "AATSC8m",
    "AATSC0v",
    "AATSC1v",
    "AATSC2v",
    "AATSC3v",
    "AATSC4v",
    "AATSC5v",
    "AATSC6v",
    "AATSC7v",
    "AATSC8v",
    "AATSC0se",
    "AATSC1se",
    "AATSC2se",
    "AATSC3se",
    "AATSC4se",
    "AATSC5se",
    "AATSC6se",
    "AATSC7se",
    "AATSC8se",
    "AATSC0pe",
    "AATSC1pe",
    "AATSC2pe",
    "AATSC3pe",
    "AATSC4pe",
    "AATSC5pe",
    "AATSC6pe",
    "AATSC7pe",
    "AATSC8pe",
    "AATSC0are",
    "AATSC1are",
    "AATSC2are",
    "AATSC3are",
    "AATSC4are",
    "AATSC5are",
    "AATSC6are",
    "AATSC7are",
    "AATSC8are",
    "AATSC0p",
    "AATSC1p",
    "AATSC2p",
    "AATSC3p",
    "AATSC4p",
    "AATSC5p",
    "AATSC6p",
    "AATSC7p",
    "AATSC8p",
    "AATSC0i",
    "AATSC1i",
    "AATSC2i",
    "AATSC3i",
    "AATSC4i",
    "AATSC5i",
    "AATSC6i",
    "AATSC7i",
    "AATSC8i",
    "MATS1c",
    "MATS2c",
    "MATS3c",
    "MATS4c",
    "MATS5c",
    "MATS6c",
    "MATS7c",
    "MATS8c",
    "MATS1dv",
    "MATS2dv",
    "MATS3dv",
    "MATS4dv",
    "MATS5dv",
    "MATS6dv",
    "MATS7dv",
    "MATS8dv",
    "MATS1d",
    "MATS2d",
    "MATS3d",
    "MATS4d",
    "MATS5d",
    "MATS6d",
    "MATS7d",
    "MATS8d",
    "MATS1s",
    "MATS2s",
    "MATS3s",
    "MATS4s",
    "MATS5s",
    "MATS6s",
    "MATS7s",
    "MATS8s",
    "MATS1Z",
    "MATS2Z",
    "MATS3Z",
    "MATS4Z",
    "MATS5Z",
    "MATS6Z",
    "MATS7Z",
    "MATS8Z",
    "MATS1m",
    "MATS2m",
    "MATS3m",
    "MATS4m",
    "MATS5m",
    "MATS6m",
    "MATS7m",
    "MATS8m",
    "MATS1v",
    "MATS2v",
    "MATS3v",
    "MATS4v",
    "MATS5v",
    "MATS6v",
    "MATS7v",
    "MATS8v",
    "MATS1se",
    "MATS2se",
    "MATS3se",
    "MATS4se",
    "MATS5se",
    "MATS6se",
    "MATS7se",
    "MATS8se",
    "MATS1pe",
    "MATS2pe",
    "MATS3pe",
    "MATS4pe",
    "MATS5pe",
    "MATS6pe",
    "MATS7pe",
    "MATS8pe",
    "MATS1are",
    "MATS2are",
    "MATS3are",
    "MATS4are",
    "MATS5are",
    "MATS6are",
    "MATS7are",
    "MATS8are",
    "MATS1p",
    "MATS2p",
    "MATS3p",
    "MATS4p",
    "MATS5p",
    "MATS6p",
    "MATS7p",
    "MATS8p",
    "MATS1i",
    "MATS2i",
    "MATS3i",
    "MATS4i",
    "MATS5i",
    "MATS6i",
    "MATS7i",
    "MATS8i",
    "GATS1c",
    "GATS2c",
    "GATS3c",
    "GATS4c",
    "GATS5c",
    "GATS6c",
    "GATS7c",
    "GATS8c",
    "GATS1dv",
    "GATS2dv",
    "GATS3dv",
    "GATS4dv",
    "GATS5dv",
    "GATS6dv",
    "GATS7dv",
    "GATS8dv",
    "GATS1d",
    "GATS2d",
    "GATS3d",
    "GATS4d",
    "GATS5d",
    "GATS6d",
    "GATS7d",
    "GATS8d",
    "GATS1s",
    "GATS2s",
    "GATS3s",
    "GATS4s",
    "GATS5s",
    "GATS6s",
    "GATS7s",
    "GATS8s",
    "GATS1Z",
    "GATS2Z",
    "GATS3Z",
    "GATS4Z",
    "GATS5Z",
    "GATS6Z",
    "GATS7Z",
    "GATS8Z",
    "GATS1m",
    "GATS2m",
    "GATS3m",
    "GATS4m",
    "GATS5m",
    "GATS6m",
    "GATS7m",
    "GATS8m",
    "GATS1v",
    "GATS2v",
    "GATS3v",
    "GATS4v",
    "GATS5v",
    "GATS6v",
    "GATS7v",
    "GATS8v",
    "GATS1se",
    "GATS2se",
    "GATS3se",
    "GATS4se",
    "GATS5se",
    "GATS6se",
    "GATS7se",
    "GATS8se",
    "GATS1pe",
    "GATS2pe",
    "GATS3pe",
    "GATS4pe",
    "GATS5pe",
    "GATS6pe",
    "GATS7pe",
    "GATS8pe",
    "GATS1are",
    "GATS2are",
    "GATS3are",
    "GATS4are",
    "GATS5are",
    "GATS6are",
    "GATS7are",
    "GATS8are",
    "GATS1p",
    "GATS2p",
    "GATS3p",
    "GATS4p",
    "GATS5p",
    "GATS6p",
    "GATS7p",
    "GATS8p",
    "GATS1i",
    "GATS2i",
    "GATS3i",
    "GATS4i",
    "GATS5i",
    "GATS6i",
    "GATS7i",
    "GATS8i",
)
