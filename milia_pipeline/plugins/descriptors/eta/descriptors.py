"""
Extended Topochemical Atom (ETA) descriptors for MILIA
(descriptor programme, Pace 8 / v1.11.0).

First-party PLUGIN-NATIVE descriptors: the second-generation ETA indices of Roy & Ghosh
(Roy & Ghosh, *QSAR Comb. Sci.* 2003, doi:10.1002/qsar.200390007; *J. Hazard. Mater.* 2013,
doi:10.1016/j.jhazmat.2013.03.023). Computed RDKit-natively (RDKit graph + the published
per-atom ETA primitives) and validated bit-exact against the ``mordredcommunity`` oracle
(rel-err 0; 45/45 on the panel). Zero-core-modification plugin contract (``name`` ==
``function_name``; NaN on any failure, never raise).

Family (45): core count (alpha), shape indices, valence-electron-mobile counts (beta:
sigma / nonsigma / delta), composite index (eta: local / reference / functionality /
branching), delta-alpha, epsilon (5 types), delta-epsilon, delta-beta, psi and delta-psi,
each in plain and averaged (``AETA_*``) forms as defined by the oracle.

Graphs: all descriptors use the kekulized, hydrogen-suppressed molecular graph (aromatic
flags preserved). Reference (all-carbon, single-bond) and saturated skeletons are built for
the reference/functionality/branching and epsilon-3/4 terms exactly as the oracle does.
Disconnected or empty (0-atom) molecules return NaN for all 45 (oracle require_connected).

Author: Asadollah (Shahram) Boshra
License: MIT
"""

from __future__ import annotations

import functools
import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

logger = logging.getLogger(__name__)
_NAN = float("nan")

_PT = Chem.GetPeriodicTable()
_PERIOD_BOUNDS = (2, 10, 18, 36, 54, 86, 118)


def _period(z: int) -> int:
    for i, b in enumerate(_PERIOD_BOUNDS):
        if z <= b:
            return i + 1
    return 7


# ---- per-atom ETA primitives (Roy-Ghosh; oracle-faithful) ----
def _core_count(a):
    z = a.GetAtomicNum()
    if z == 1:
        return 0.0
    zv = _PT.GetNOuterElecs(z)
    pn = _period(z)
    return (z - zv) / (zv * (pn - 1))


def _epsilon(a):
    zv = _PT.GetNOuterElecs(a.GetAtomicNum())
    return 0.3 * zv - _core_count(a)


def _beta_sigma(a):
    e = _epsilon(a)
    return sum(
        0.5 if abs(_epsilon(n) - e) <= 0.3 else 0.75
        for n in a.GetNeighbors()
        if n.GetAtomicNum() != 1
    )


def _nonsigma_contribute(bond):
    if bond.GetBondType() is Chem.BondType.SINGLE:
        return 0.0
    f = 1.0
    if bond.GetBondTypeAsDouble() == Chem.BondType.TRIPLE:
        f = 2.0
    d_eps = abs(_epsilon(bond.GetBeginAtom()) - _epsilon(bond.GetEndAtom()))
    if bond.GetIsAromatic():
        y = 2.0
    elif d_eps > 0.3:
        y = 1.5
    else:
        y = 1.0
    return y * f


def _beta_delta(a):
    if (
        a.GetIsAromatic()
        or a.IsInRing()
        or _PT.GetNOuterElecs(a.GetAtomicNum()) - a.GetTotalValence() <= 0
    ):
        return 0.0
    for b in a.GetNeighbors():
        if b.GetIsAromatic():
            return 0.5
    return 0.0


def _other(bond, atom):
    beg = bond.GetBeginAtom()
    return beg if atom.GetIdx() != beg.GetIdx() else bond.GetEndAtom()


def _beta_non_sigma(a):
    return sum(_nonsigma_contribute(b) for b in a.GetBonds() if _other(b, a).GetAtomicNum() != 1)


def _gamma(a):
    beta = _beta_sigma(a) + _beta_non_sigma(a) + _beta_delta(a)
    if beta == 0:
        return _NAN
    return _core_count(a) / beta


# ---- molecule variants ----
def _prep(mol, explicit_h):
    m = Chem.AddHs(mol) if explicit_h else Chem.RemoveHs(mol)
    Chem.Kekulize(m)
    return m


def _alter(mol, explicit_h, saturated):
    src = Chem.RemoveHs(mol)
    Chem.Kekulize(src)
    new = Chem.RWMol(Chem.Mol())
    ids = {}
    for a in src.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        if saturated:
            na = Chem.Atom(a.GetAtomicNum())
            na.SetFormalCharge(a.GetFormalCharge())
        else:
            na = Chem.Atom(6)
        ids[a.GetIdx()] = new.AddAtom(na)
    for bond in src.GetBonds():
        ai, aj = bond.GetBeginAtom(), bond.GetEndAtom()
        if not saturated and (ai.GetDegree() > 4 or aj.GetDegree() > 4):
            raise ValueError("bond degree greater than 4")
        i, j = ids.get(ai.GetIdx()), ids.get(aj.GetIdx())
        if i is not None and j is not None:
            if saturated and (ai.GetAtomicNum() != 6 or aj.GetAtomicNum() != 6):
                order = bond.GetBondType()
            else:
                order = Chem.BondType.SINGLE
            new.AddBond(i, j, order)
    new = Chem.Mol(new)
    if Chem.SanitizeMol(new, catchErrors=True) != 0:
        raise ValueError("cannot sanitize altered molecule")
    if explicit_h:
        new = Chem.AddHs(new)
    Chem.Kekulize(new)
    return new


def _alpha(mol, averaged):
    v = sum(_core_count(a) for a in mol.GetAtoms())
    return v / mol.GetNumAtoms() if averaged else v


def _eta_composite(mol, local, averaged):
    d = Chem.GetDistanceMatrix(mol, force=True)
    gamma = np.array([_gamma(a) for a in mol.GetAtoms()])
    if local:

        def chk(r):
            return r == 1
    else:

        def chk(r):
            return r != 0

    v = sum(
        sum(np.sqrt(gamma[i] * gamma[j] / r**2) for j, r in enumerate(row) if i < j and chk(r))
        for i, row in enumerate(d)
    )
    return v / mol.GetNumAtoms() if averaged else v


def _epsilon_val(mol, base, t):
    if t == 2:
        return sum(_epsilon(a) for a in base.GetAtoms()) / base.GetNumAtoms()
    if t == 1:
        m = _prep(mol, True)
    elif t == 3:
        m = _alter(mol, True, False)
    elif t == 4:
        m = _alter(mol, True, True)
    else:  # t == 5
        m = _prep(mol, True)
        eps = [
            _epsilon(a)
            for a in m.GetAtoms()
            if a.GetAtomicNum() != 1 or a.GetNeighbors()[0].GetAtomicNum() != 6
        ]
        return sum(eps) / len(eps)
    return sum(_epsilon(a) for a in m.GetAtoms()) / m.GetNumAtoms()


@functools.lru_cache(maxsize=1)
def _eta_block(mol):
    """Compute all 45 ETA descriptors once per molecule (single-slot memoised).

    Disconnected or 0-atom molecules -> NaN for all (oracle require_connected).
    """
    out = {name: _NAN for name in ALL_DESCRIPTOR_NAMES}
    base = _prep(mol, False)
    n = base.GetNumAtoms()
    if n == 0 or len(Chem.GetMolFrags(base)) != 1:
        return out

    def _safe(fn):
        try:
            return fn()
        except Exception:
            return _NAN

    # alpha + shape
    a_full = _alpha(base, False)
    out["ETA_alpha"] = a_full
    out["AETA_alpha"] = a_full / n
    for t, deg in (("p", 1), ("y", 3), ("x", 4)):
        out["ETA_shape_" + t] = (
            sum(_core_count(a) for a in base.GetAtoms() if a.GetDegree() == deg) / a_full
        )
    # beta family
    b_s = sum(_beta_sigma(a) / 2.0 for a in base.GetAtoms())
    b_nsd = sum(_beta_delta(a) for a in base.GetAtoms())
    b_ns = sum(_beta_non_sigma(a) / 2.0 + _beta_delta(a) for a in base.GetAtoms())
    b_all = sum(
        _beta_sigma(a) / 2.0 + _beta_non_sigma(a) / 2.0 + _beta_delta(a) for a in base.GetAtoms()
    )
    for typ, val in (("", b_all), ("s", b_s), ("ns", b_ns), ("ns_d", b_nsd)):
        suff = ("_" + typ) if typ else ""
        out["ETA_beta" + suff] = val
        out["AETA_beta" + suff] = val / n
    # reference graph
    try:
        ref = _alter(mol, False, False)
    except Exception:
        ref = None
    # eta composite (non-ref) + reference
    for local in (False, True):
        v = _eta_composite(base, local, False)
        out["ETA_eta" + ("_L" if local else "")] = v
        out["AETA_eta" + ("_L" if local else "")] = v / n
        vr = _NAN if ref is None else _eta_composite(ref, local, False)
        rsuff = "_R" + ("L" if local else "")
        out["ETA_eta" + rsuff] = vr
        out["AETA_eta" + rsuff] = vr / n
    # functionality: eta_R - eta
    for local in (False, True):
        eta = _eta_composite(base, local, False)
        eta_r = _NAN if ref is None else _eta_composite(ref, local, False)
        v = eta_r - eta
        suff = "L" if local else ""
        out["ETA_eta_F" + suff] = v
        out["AETA_eta_F" + suff] = v / n
    # branching
    nring = rdMolDescriptors.CalcNumRings(Chem.RemoveHs(mol))
    eta_rl = _NAN if ref is None else _eta_composite(ref, True, False)
    if n <= 1:
        eta_nl = _NAN
    elif n == 2:
        eta_nl = 1.0
    else:
        eta_nl = np.sqrt(2) + 0.5 * (n - 3)
    for ring in (False, True):
        v = eta_nl - eta_rl + 0.086 * (nring if ring else 0)
        suff = "R" if ring else ""
        out["ETA_eta_B" + suff] = v
        out["AETA_eta_B" + suff] = v / n
    # delta alpha
    alpha_r = _NAN if ref is None else _alpha(ref, False)
    out["ETA_dAlpha_A"] = max((a_full - alpha_r) / n, 0.0) if alpha_r == alpha_r else _NAN
    out["ETA_dAlpha_B"] = max((alpha_r - a_full) / n, 0.0) if alpha_r == alpha_r else _NAN
    # epsilon (5 types)
    eps = {t: _safe(lambda t=t: _epsilon_val(mol, base, t)) for t in range(1, 6)}
    for t in range(1, 6):
        out[f"ETA_epsilon_{t}"] = eps[t]
    for typ, (lft, rgt) in {"A": (1, 3), "B": (1, 4), "C": (3, 4), "D": (2, 5)}.items():
        out["ETA_dEpsilon_" + typ] = eps[lft] - eps[rgt]
    # delta beta
    db = b_ns - b_s
    out["ETA_dBeta"] = db
    out["AETA_dBeta"] = db / n
    # psi + delta psi
    psi = a_full / (n * eps[2]) if eps[2] == eps[2] else _NAN
    out["ETA_psi_1"] = psi
    out["ETA_dPsi_A"] = max(0.714 - psi, 0.0) if psi == psi else _NAN
    out["ETA_dPsi_B"] = max(psi - 0.714, 0.0) if psi == psi else _NAN
    return out


def _make(name):
    def fn(mol):
        try:
            if mol is None:
                return _NAN
            v = _eta_block(mol)[name]
            return float(v)
        except Exception:
            return _NAN

    return fn


_ETA_alpha = _make("ETA_alpha")
_AETA_alpha = _make("AETA_alpha")
_ETA_shape_p = _make("ETA_shape_p")
_ETA_shape_y = _make("ETA_shape_y")
_ETA_shape_x = _make("ETA_shape_x")
_ETA_beta = _make("ETA_beta")
_AETA_beta = _make("AETA_beta")
_ETA_beta_s = _make("ETA_beta_s")
_AETA_beta_s = _make("AETA_beta_s")
_ETA_beta_ns = _make("ETA_beta_ns")
_AETA_beta_ns = _make("AETA_beta_ns")
_ETA_beta_ns_d = _make("ETA_beta_ns_d")
_AETA_beta_ns_d = _make("AETA_beta_ns_d")
_ETA_eta = _make("ETA_eta")
_AETA_eta = _make("AETA_eta")
_ETA_eta_L = _make("ETA_eta_L")
_AETA_eta_L = _make("AETA_eta_L")
_ETA_eta_R = _make("ETA_eta_R")
_AETA_eta_R = _make("AETA_eta_R")
_ETA_eta_RL = _make("ETA_eta_RL")
_AETA_eta_RL = _make("AETA_eta_RL")
_ETA_eta_F = _make("ETA_eta_F")
_AETA_eta_F = _make("AETA_eta_F")
_ETA_eta_FL = _make("ETA_eta_FL")
_AETA_eta_FL = _make("AETA_eta_FL")
_ETA_eta_B = _make("ETA_eta_B")
_AETA_eta_B = _make("AETA_eta_B")
_ETA_eta_BR = _make("ETA_eta_BR")
_AETA_eta_BR = _make("AETA_eta_BR")
_ETA_dAlpha_A = _make("ETA_dAlpha_A")
_ETA_dAlpha_B = _make("ETA_dAlpha_B")
_ETA_epsilon_1 = _make("ETA_epsilon_1")
_ETA_epsilon_2 = _make("ETA_epsilon_2")
_ETA_epsilon_3 = _make("ETA_epsilon_3")
_ETA_epsilon_4 = _make("ETA_epsilon_4")
_ETA_epsilon_5 = _make("ETA_epsilon_5")
_ETA_dEpsilon_A = _make("ETA_dEpsilon_A")
_ETA_dEpsilon_B = _make("ETA_dEpsilon_B")
_ETA_dEpsilon_C = _make("ETA_dEpsilon_C")
_ETA_dEpsilon_D = _make("ETA_dEpsilon_D")
_ETA_dBeta = _make("ETA_dBeta")
_AETA_dBeta = _make("AETA_dBeta")
_ETA_psi_1 = _make("ETA_psi_1")
_ETA_dPsi_A = _make("ETA_dPsi_A")
_ETA_dPsi_B = _make("ETA_dPsi_B")


# ---- public wrappers (name == function_name; declared -> 0 bonus) ----


def ETA_alpha(mol):
    """ETA core count (alpha)."""
    return _ETA_alpha(mol)


def AETA_alpha(mol):
    """averaged ETA core count."""
    return _AETA_alpha(mol)


def ETA_shape_p(mol):
    """ETA shape index p (terminal)."""
    return _ETA_shape_p(mol)


def ETA_shape_y(mol):
    """ETA shape index y (degree-3)."""
    return _ETA_shape_y(mol)


def ETA_shape_x(mol):
    """ETA shape index x (degree-4)."""
    return _ETA_shape_x(mol)


def ETA_beta(mol):
    """ETA valence electron mobile count (beta)."""
    return _ETA_beta(mol)


def AETA_beta(mol):
    """averaged ETA VEM count."""
    return _AETA_beta(mol)


def ETA_beta_s(mol):
    """ETA sigma VEM count."""
    return _ETA_beta_s(mol)


def AETA_beta_s(mol):
    """averaged ETA sigma VEM count."""
    return _AETA_beta_s(mol)


def ETA_beta_ns(mol):
    """ETA nonsigma VEM count."""
    return _ETA_beta_ns(mol)


def AETA_beta_ns(mol):
    """averaged ETA nonsigma VEM count."""
    return _AETA_beta_ns(mol)


def ETA_beta_ns_d(mol):
    """ETA delta VEM count."""
    return _ETA_beta_ns_d(mol)


def AETA_beta_ns_d(mol):
    """averaged ETA delta VEM count."""
    return _AETA_beta_ns_d(mol)


def ETA_eta(mol):
    """ETA composite index."""
    return _ETA_eta(mol)


def AETA_eta(mol):
    """averaged ETA composite index."""
    return _AETA_eta(mol)


def ETA_eta_L(mol):
    """local ETA composite index."""
    return _ETA_eta_L(mol)


def AETA_eta_L(mol):
    """averaged local ETA composite index."""
    return _AETA_eta_L(mol)


def ETA_eta_R(mol):
    """reference ETA composite index."""
    return _ETA_eta_R(mol)


def AETA_eta_R(mol):
    """averaged reference ETA composite index."""
    return _AETA_eta_R(mol)


def ETA_eta_RL(mol):
    """local reference ETA composite index."""
    return _ETA_eta_RL(mol)


def AETA_eta_RL(mol):
    """averaged local reference ETA composite index."""
    return _AETA_eta_RL(mol)


def ETA_eta_F(mol):
    """ETA functionality index."""
    return _ETA_eta_F(mol)


def AETA_eta_F(mol):
    """averaged ETA functionality index."""
    return _AETA_eta_F(mol)


def ETA_eta_FL(mol):
    """local ETA functionality index."""
    return _ETA_eta_FL(mol)


def AETA_eta_FL(mol):
    """averaged local ETA functionality index."""
    return _AETA_eta_FL(mol)


def ETA_eta_B(mol):
    """ETA branching index."""
    return _ETA_eta_B(mol)


def AETA_eta_B(mol):
    """averaged ETA branching index."""
    return _AETA_eta_B(mol)


def ETA_eta_BR(mol):
    """ETA branching index (ring-corrected)."""
    return _ETA_eta_BR(mol)


def AETA_eta_BR(mol):
    """averaged ETA branching index (ring-corrected)."""
    return _AETA_eta_BR(mol)


def ETA_dAlpha_A(mol):
    """ETA delta alpha A."""
    return _ETA_dAlpha_A(mol)


def ETA_dAlpha_B(mol):
    """ETA delta alpha B."""
    return _ETA_dAlpha_B(mol)


def ETA_epsilon_1(mol):
    """ETA epsilon (all atoms)."""
    return _ETA_epsilon_1(mol)


def ETA_epsilon_2(mol):
    """ETA epsilon (heavy atoms)."""
    return _ETA_epsilon_2(mol)


def ETA_epsilon_3(mol):
    """ETA epsilon (reference alkane)."""
    return _ETA_epsilon_3(mol)


def ETA_epsilon_4(mol):
    """ETA epsilon (saturated skeleton)."""
    return _ETA_epsilon_4(mol)


def ETA_epsilon_5(mol):
    """ETA epsilon (heavy + hetero-H)."""
    return _ETA_epsilon_5(mol)


def ETA_dEpsilon_A(mol):
    """ETA delta epsilon A (eps1-eps3)."""
    return _ETA_dEpsilon_A(mol)


def ETA_dEpsilon_B(mol):
    """ETA delta epsilon B (eps1-eps4)."""
    return _ETA_dEpsilon_B(mol)


def ETA_dEpsilon_C(mol):
    """ETA delta epsilon C (eps3-eps4)."""
    return _ETA_dEpsilon_C(mol)


def ETA_dEpsilon_D(mol):
    """ETA delta epsilon D (eps2-eps5)."""
    return _ETA_dEpsilon_D(mol)


def ETA_dBeta(mol):
    """ETA delta beta (nonsigma-sigma)."""
    return _ETA_dBeta(mol)


def AETA_dBeta(mol):
    """averaged ETA delta beta."""
    return _AETA_dBeta(mol)


def ETA_psi_1(mol):
    """ETA psi."""
    return _ETA_psi_1(mol)


def ETA_dPsi_A(mol):
    """ETA delta psi A."""
    return _ETA_dPsi_A(mol)


def ETA_dPsi_B(mol):
    """ETA delta psi B."""
    return _ETA_dPsi_B(mol)


ALL_DESCRIPTOR_NAMES = (
    "ETA_alpha",
    "AETA_alpha",
    "ETA_shape_p",
    "ETA_shape_y",
    "ETA_shape_x",
    "ETA_beta",
    "AETA_beta",
    "ETA_beta_s",
    "AETA_beta_s",
    "ETA_beta_ns",
    "AETA_beta_ns",
    "ETA_beta_ns_d",
    "AETA_beta_ns_d",
    "ETA_eta",
    "AETA_eta",
    "ETA_eta_L",
    "AETA_eta_L",
    "ETA_eta_R",
    "AETA_eta_R",
    "ETA_eta_RL",
    "AETA_eta_RL",
    "ETA_eta_F",
    "AETA_eta_F",
    "ETA_eta_FL",
    "AETA_eta_FL",
    "ETA_eta_B",
    "AETA_eta_B",
    "ETA_eta_BR",
    "AETA_eta_BR",
    "ETA_dAlpha_A",
    "ETA_dAlpha_B",
    "ETA_epsilon_1",
    "ETA_epsilon_2",
    "ETA_epsilon_3",
    "ETA_epsilon_4",
    "ETA_epsilon_5",
    "ETA_dEpsilon_A",
    "ETA_dEpsilon_B",
    "ETA_dEpsilon_C",
    "ETA_dEpsilon_D",
    "ETA_dBeta",
    "AETA_dBeta",
    "ETA_psi_1",
    "ETA_dPsi_A",
    "ETA_dPsi_B",
)
