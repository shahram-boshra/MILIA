"""
Kier-Hall electrotopological-state (E-State) atom-type descriptors for MILIA
(descriptor programme, Pace 7 / v1.10.0).

First-party PLUGIN-NATIVE descriptors computed RDKit-natively (Kier-Hall intrinsic +
field E-State on the hydrogen-suppressed graph) and validated bit-close against the
``mordredcommunity`` oracle (rel-err 0; 316/316). Zero-core-modification plugin contract
(``name`` == ``function_name``; NaN on any failure, never raise).

Family (316) = 79 Kier-Hall atom types x 4 aggregations:
    N<type>   count  of atoms of the E-State atom type
    S<type>   sum    of E-State values over atoms of that type
    MAX<type> max    E-State value among atoms of that type (NaN if none)
    MIN<type> min    E-State value among atoms of that type (NaN if none)

Primary ref: Kier & Hall, *Molecular Structure Description* (1999); Hall & Kier,
J. Chem. Inf. Comput. Sci. 1995, 35, 1039. Atom typing + E-State indices from RDKit
(``rdkit.Chem.EState``), whose 79-label vocabulary is identical to the mordred oracle
(verified: empty symmetric difference). Absent-type convention matches the oracle
(N=0, S=0, MAX=NaN, MIN=NaN).

Author: Asadollah (Shahram) Boshra
License: MIT
"""

from __future__ import annotations

import functools
import logging

from rdkit import Chem
from rdkit.Chem.EState import AtomTypes as _AtomTypes
from rdkit.Chem.EState import EStateIndices as _EStateIndices

logger = logging.getLogger(__name__)
_NAN = float("nan")

# 79 Kier-Hall E-State atom-type labels (authoritative mordred/RDKit order).
_ES_TYPES: tuple[str, ...] = (
    "sLi",
    "ssBe",
    "ssssBe",
    "ssBH",
    "sssB",
    "ssssB",
    "sCH3",
    "dCH2",
    "ssCH2",
    "tCH",
    "dsCH",
    "aaCH",
    "sssCH",
    "ddC",
    "tsC",
    "dssC",
    "aasC",
    "aaaC",
    "ssssC",
    "sNH3",
    "sNH2",
    "ssNH2",
    "dNH",
    "ssNH",
    "aaNH",
    "tN",
    "sssNH",
    "dsN",
    "aaN",
    "sssN",
    "ddsN",
    "aasN",
    "ssssN",
    "sOH",
    "dO",
    "ssO",
    "aaO",
    "sF",
    "sSiH3",
    "ssSiH2",
    "sssSiH",
    "ssssSi",
    "sPH2",
    "ssPH",
    "sssP",
    "dsssP",
    "sssssP",
    "sSH",
    "dS",
    "ssS",
    "aaS",
    "dssS",
    "ddssS",
    "sCl",
    "sGeH3",
    "ssGeH2",
    "sssGeH",
    "ssssGe",
    "sAsH2",
    "ssAsH",
    "sssAs",
    "sssdAs",
    "sssssAs",
    "sSeH",
    "dSe",
    "ssSe",
    "aaSe",
    "dssSe",
    "ddssSe",
    "sBr",
    "sSnH3",
    "ssSnH2",
    "sssSnH",
    "ssssSn",
    "sI",
    "sPbH3",
    "ssPbH2",
    "sssPbH",
    "ssssPb",
)


@functools.lru_cache(maxsize=1)
def _estate_block(mol: Chem.Mol) -> dict[str, tuple[float, float, float, float]]:
    """Compute (count, sum, max, min) of E-State values per atom type, once per mol.

    Memoised single-slot (maxsize=1) keyed on the RDKit ``Mol`` object: the calculator
    passes one ``mol`` to every descriptor wrapper, so the block is computed once and the
    remaining 315 wrappers hit the cache; the next molecule evicts the single slot.
    Absent types are reported as (0, 0.0, NaN, NaN) to match the oracle.
    """
    per_type: dict[str, list[float]] = {t: [] for t in _ES_TYPES}
    if mol.GetNumAtoms() == 0:
        # Empty / atomless molecule is invalid -> NaN for every aggregation
        # (MILIA purity contract: invalid molecule -> NaN; oracle is internally
        # inconsistent here, returning NaN for counts but 0 for sums).
        return {t: (_NAN, _NAN, _NAN, _NAN) for t in _ES_TYPES}
    atom_types = _AtomTypes.TypeAtoms(mol)  # per-atom tuple of Kier-Hall labels
    estates = _EStateIndices(mol)  # per-atom E-State value (aligned)
    for labels, value in zip(atom_types, estates, strict=False):
        for label in labels:
            bucket = per_type.get(label)
            if bucket is not None:
                bucket.append(float(value))
    out: dict[str, tuple[float, float, float, float]] = {}
    for t, vals in per_type.items():
        if vals:
            out[t] = (float(len(vals)), float(sum(vals)), float(max(vals)), float(min(vals)))
        else:
            out[t] = (0.0, 0.0, _NAN, _NAN)
    return out


# aggregation index into the per-type tuple
_AGGR_IDX = {"N": 0, "S": 1, "MAX": 2, "MIN": 3}


def _make(aggr: str, es_type: str):
    idx = _AGGR_IDX[aggr]

    def fn(mol):
        try:
            if mol is None:
                return _NAN
            return _estate_block(mol)[es_type][idx]
        except Exception:
            return _NAN

    return fn


_NsLi = _make("N", "sLi")
_NssBe = _make("N", "ssBe")
_NssssBe = _make("N", "ssssBe")
_NssBH = _make("N", "ssBH")
_NsssB = _make("N", "sssB")
_NssssB = _make("N", "ssssB")
_NsCH3 = _make("N", "sCH3")
_NdCH2 = _make("N", "dCH2")
_NssCH2 = _make("N", "ssCH2")
_NtCH = _make("N", "tCH")
_NdsCH = _make("N", "dsCH")
_NaaCH = _make("N", "aaCH")
_NsssCH = _make("N", "sssCH")
_NddC = _make("N", "ddC")
_NtsC = _make("N", "tsC")
_NdssC = _make("N", "dssC")
_NaasC = _make("N", "aasC")
_NaaaC = _make("N", "aaaC")
_NssssC = _make("N", "ssssC")
_NsNH3 = _make("N", "sNH3")
_NsNH2 = _make("N", "sNH2")
_NssNH2 = _make("N", "ssNH2")
_NdNH = _make("N", "dNH")
_NssNH = _make("N", "ssNH")
_NaaNH = _make("N", "aaNH")
_NtN = _make("N", "tN")
_NsssNH = _make("N", "sssNH")
_NdsN = _make("N", "dsN")
_NaaN = _make("N", "aaN")
_NsssN = _make("N", "sssN")
_NddsN = _make("N", "ddsN")
_NaasN = _make("N", "aasN")
_NssssN = _make("N", "ssssN")
_NsOH = _make("N", "sOH")
_NdO = _make("N", "dO")
_NssO = _make("N", "ssO")
_NaaO = _make("N", "aaO")
_NsF = _make("N", "sF")
_NsSiH3 = _make("N", "sSiH3")
_NssSiH2 = _make("N", "ssSiH2")
_NsssSiH = _make("N", "sssSiH")
_NssssSi = _make("N", "ssssSi")
_NsPH2 = _make("N", "sPH2")
_NssPH = _make("N", "ssPH")
_NsssP = _make("N", "sssP")
_NdsssP = _make("N", "dsssP")
_NsssssP = _make("N", "sssssP")
_NsSH = _make("N", "sSH")
_NdS = _make("N", "dS")
_NssS = _make("N", "ssS")
_NaaS = _make("N", "aaS")
_NdssS = _make("N", "dssS")
_NddssS = _make("N", "ddssS")
_NsCl = _make("N", "sCl")
_NsGeH3 = _make("N", "sGeH3")
_NssGeH2 = _make("N", "ssGeH2")
_NsssGeH = _make("N", "sssGeH")
_NssssGe = _make("N", "ssssGe")
_NsAsH2 = _make("N", "sAsH2")
_NssAsH = _make("N", "ssAsH")
_NsssAs = _make("N", "sssAs")
_NsssdAs = _make("N", "sssdAs")
_NsssssAs = _make("N", "sssssAs")
_NsSeH = _make("N", "sSeH")
_NdSe = _make("N", "dSe")
_NssSe = _make("N", "ssSe")
_NaaSe = _make("N", "aaSe")
_NdssSe = _make("N", "dssSe")
_NddssSe = _make("N", "ddssSe")
_NsBr = _make("N", "sBr")
_NsSnH3 = _make("N", "sSnH3")
_NssSnH2 = _make("N", "ssSnH2")
_NsssSnH = _make("N", "sssSnH")
_NssssSn = _make("N", "ssssSn")
_NsI = _make("N", "sI")
_NsPbH3 = _make("N", "sPbH3")
_NssPbH2 = _make("N", "ssPbH2")
_NsssPbH = _make("N", "sssPbH")
_NssssPb = _make("N", "ssssPb")
_SsLi = _make("S", "sLi")
_SssBe = _make("S", "ssBe")
_SssssBe = _make("S", "ssssBe")
_SssBH = _make("S", "ssBH")
_SsssB = _make("S", "sssB")
_SssssB = _make("S", "ssssB")
_SsCH3 = _make("S", "sCH3")
_SdCH2 = _make("S", "dCH2")
_SssCH2 = _make("S", "ssCH2")
_StCH = _make("S", "tCH")
_SdsCH = _make("S", "dsCH")
_SaaCH = _make("S", "aaCH")
_SsssCH = _make("S", "sssCH")
_SddC = _make("S", "ddC")
_StsC = _make("S", "tsC")
_SdssC = _make("S", "dssC")
_SaasC = _make("S", "aasC")
_SaaaC = _make("S", "aaaC")
_SssssC = _make("S", "ssssC")
_SsNH3 = _make("S", "sNH3")
_SsNH2 = _make("S", "sNH2")
_SssNH2 = _make("S", "ssNH2")
_SdNH = _make("S", "dNH")
_SssNH = _make("S", "ssNH")
_SaaNH = _make("S", "aaNH")
_StN = _make("S", "tN")
_SsssNH = _make("S", "sssNH")
_SdsN = _make("S", "dsN")
_SaaN = _make("S", "aaN")
_SsssN = _make("S", "sssN")
_SddsN = _make("S", "ddsN")
_SaasN = _make("S", "aasN")
_SssssN = _make("S", "ssssN")
_SsOH = _make("S", "sOH")
_SdO = _make("S", "dO")
_SssO = _make("S", "ssO")
_SaaO = _make("S", "aaO")
_SsF = _make("S", "sF")
_SsSiH3 = _make("S", "sSiH3")
_SssSiH2 = _make("S", "ssSiH2")
_SsssSiH = _make("S", "sssSiH")
_SssssSi = _make("S", "ssssSi")
_SsPH2 = _make("S", "sPH2")
_SssPH = _make("S", "ssPH")
_SsssP = _make("S", "sssP")
_SdsssP = _make("S", "dsssP")
_SsssssP = _make("S", "sssssP")
_SsSH = _make("S", "sSH")
_SdS = _make("S", "dS")
_SssS = _make("S", "ssS")
_SaaS = _make("S", "aaS")
_SdssS = _make("S", "dssS")
_SddssS = _make("S", "ddssS")
_SsCl = _make("S", "sCl")
_SsGeH3 = _make("S", "sGeH3")
_SssGeH2 = _make("S", "ssGeH2")
_SsssGeH = _make("S", "sssGeH")
_SssssGe = _make("S", "ssssGe")
_SsAsH2 = _make("S", "sAsH2")
_SssAsH = _make("S", "ssAsH")
_SsssAs = _make("S", "sssAs")
_SsssdAs = _make("S", "sssdAs")
_SsssssAs = _make("S", "sssssAs")
_SsSeH = _make("S", "sSeH")
_SdSe = _make("S", "dSe")
_SssSe = _make("S", "ssSe")
_SaaSe = _make("S", "aaSe")
_SdssSe = _make("S", "dssSe")
_SddssSe = _make("S", "ddssSe")
_SsBr = _make("S", "sBr")
_SsSnH3 = _make("S", "sSnH3")
_SssSnH2 = _make("S", "ssSnH2")
_SsssSnH = _make("S", "sssSnH")
_SssssSn = _make("S", "ssssSn")
_SsI = _make("S", "sI")
_SsPbH3 = _make("S", "sPbH3")
_SssPbH2 = _make("S", "ssPbH2")
_SsssPbH = _make("S", "sssPbH")
_SssssPb = _make("S", "ssssPb")
_MAXsLi = _make("MAX", "sLi")
_MAXssBe = _make("MAX", "ssBe")
_MAXssssBe = _make("MAX", "ssssBe")
_MAXssBH = _make("MAX", "ssBH")
_MAXsssB = _make("MAX", "sssB")
_MAXssssB = _make("MAX", "ssssB")
_MAXsCH3 = _make("MAX", "sCH3")
_MAXdCH2 = _make("MAX", "dCH2")
_MAXssCH2 = _make("MAX", "ssCH2")
_MAXtCH = _make("MAX", "tCH")
_MAXdsCH = _make("MAX", "dsCH")
_MAXaaCH = _make("MAX", "aaCH")
_MAXsssCH = _make("MAX", "sssCH")
_MAXddC = _make("MAX", "ddC")
_MAXtsC = _make("MAX", "tsC")
_MAXdssC = _make("MAX", "dssC")
_MAXaasC = _make("MAX", "aasC")
_MAXaaaC = _make("MAX", "aaaC")
_MAXssssC = _make("MAX", "ssssC")
_MAXsNH3 = _make("MAX", "sNH3")
_MAXsNH2 = _make("MAX", "sNH2")
_MAXssNH2 = _make("MAX", "ssNH2")
_MAXdNH = _make("MAX", "dNH")
_MAXssNH = _make("MAX", "ssNH")
_MAXaaNH = _make("MAX", "aaNH")
_MAXtN = _make("MAX", "tN")
_MAXsssNH = _make("MAX", "sssNH")
_MAXdsN = _make("MAX", "dsN")
_MAXaaN = _make("MAX", "aaN")
_MAXsssN = _make("MAX", "sssN")
_MAXddsN = _make("MAX", "ddsN")
_MAXaasN = _make("MAX", "aasN")
_MAXssssN = _make("MAX", "ssssN")
_MAXsOH = _make("MAX", "sOH")
_MAXdO = _make("MAX", "dO")
_MAXssO = _make("MAX", "ssO")
_MAXaaO = _make("MAX", "aaO")
_MAXsF = _make("MAX", "sF")
_MAXsSiH3 = _make("MAX", "sSiH3")
_MAXssSiH2 = _make("MAX", "ssSiH2")
_MAXsssSiH = _make("MAX", "sssSiH")
_MAXssssSi = _make("MAX", "ssssSi")
_MAXsPH2 = _make("MAX", "sPH2")
_MAXssPH = _make("MAX", "ssPH")
_MAXsssP = _make("MAX", "sssP")
_MAXdsssP = _make("MAX", "dsssP")
_MAXsssssP = _make("MAX", "sssssP")
_MAXsSH = _make("MAX", "sSH")
_MAXdS = _make("MAX", "dS")
_MAXssS = _make("MAX", "ssS")
_MAXaaS = _make("MAX", "aaS")
_MAXdssS = _make("MAX", "dssS")
_MAXddssS = _make("MAX", "ddssS")
_MAXsCl = _make("MAX", "sCl")
_MAXsGeH3 = _make("MAX", "sGeH3")
_MAXssGeH2 = _make("MAX", "ssGeH2")
_MAXsssGeH = _make("MAX", "sssGeH")
_MAXssssGe = _make("MAX", "ssssGe")
_MAXsAsH2 = _make("MAX", "sAsH2")
_MAXssAsH = _make("MAX", "ssAsH")
_MAXsssAs = _make("MAX", "sssAs")
_MAXsssdAs = _make("MAX", "sssdAs")
_MAXsssssAs = _make("MAX", "sssssAs")
_MAXsSeH = _make("MAX", "sSeH")
_MAXdSe = _make("MAX", "dSe")
_MAXssSe = _make("MAX", "ssSe")
_MAXaaSe = _make("MAX", "aaSe")
_MAXdssSe = _make("MAX", "dssSe")
_MAXddssSe = _make("MAX", "ddssSe")
_MAXsBr = _make("MAX", "sBr")
_MAXsSnH3 = _make("MAX", "sSnH3")
_MAXssSnH2 = _make("MAX", "ssSnH2")
_MAXsssSnH = _make("MAX", "sssSnH")
_MAXssssSn = _make("MAX", "ssssSn")
_MAXsI = _make("MAX", "sI")
_MAXsPbH3 = _make("MAX", "sPbH3")
_MAXssPbH2 = _make("MAX", "ssPbH2")
_MAXsssPbH = _make("MAX", "sssPbH")
_MAXssssPb = _make("MAX", "ssssPb")
_MINsLi = _make("MIN", "sLi")
_MINssBe = _make("MIN", "ssBe")
_MINssssBe = _make("MIN", "ssssBe")
_MINssBH = _make("MIN", "ssBH")
_MINsssB = _make("MIN", "sssB")
_MINssssB = _make("MIN", "ssssB")
_MINsCH3 = _make("MIN", "sCH3")
_MINdCH2 = _make("MIN", "dCH2")
_MINssCH2 = _make("MIN", "ssCH2")
_MINtCH = _make("MIN", "tCH")
_MINdsCH = _make("MIN", "dsCH")
_MINaaCH = _make("MIN", "aaCH")
_MINsssCH = _make("MIN", "sssCH")
_MINddC = _make("MIN", "ddC")
_MINtsC = _make("MIN", "tsC")
_MINdssC = _make("MIN", "dssC")
_MINaasC = _make("MIN", "aasC")
_MINaaaC = _make("MIN", "aaaC")
_MINssssC = _make("MIN", "ssssC")
_MINsNH3 = _make("MIN", "sNH3")
_MINsNH2 = _make("MIN", "sNH2")
_MINssNH2 = _make("MIN", "ssNH2")
_MINdNH = _make("MIN", "dNH")
_MINssNH = _make("MIN", "ssNH")
_MINaaNH = _make("MIN", "aaNH")
_MINtN = _make("MIN", "tN")
_MINsssNH = _make("MIN", "sssNH")
_MINdsN = _make("MIN", "dsN")
_MINaaN = _make("MIN", "aaN")
_MINsssN = _make("MIN", "sssN")
_MINddsN = _make("MIN", "ddsN")
_MINaasN = _make("MIN", "aasN")
_MINssssN = _make("MIN", "ssssN")
_MINsOH = _make("MIN", "sOH")
_MINdO = _make("MIN", "dO")
_MINssO = _make("MIN", "ssO")
_MINaaO = _make("MIN", "aaO")
_MINsF = _make("MIN", "sF")
_MINsSiH3 = _make("MIN", "sSiH3")
_MINssSiH2 = _make("MIN", "ssSiH2")
_MINsssSiH = _make("MIN", "sssSiH")
_MINssssSi = _make("MIN", "ssssSi")
_MINsPH2 = _make("MIN", "sPH2")
_MINssPH = _make("MIN", "ssPH")
_MINsssP = _make("MIN", "sssP")
_MINdsssP = _make("MIN", "dsssP")
_MINsssssP = _make("MIN", "sssssP")
_MINsSH = _make("MIN", "sSH")
_MINdS = _make("MIN", "dS")
_MINssS = _make("MIN", "ssS")
_MINaaS = _make("MIN", "aaS")
_MINdssS = _make("MIN", "dssS")
_MINddssS = _make("MIN", "ddssS")
_MINsCl = _make("MIN", "sCl")
_MINsGeH3 = _make("MIN", "sGeH3")
_MINssGeH2 = _make("MIN", "ssGeH2")
_MINsssGeH = _make("MIN", "sssGeH")
_MINssssGe = _make("MIN", "ssssGe")
_MINsAsH2 = _make("MIN", "sAsH2")
_MINssAsH = _make("MIN", "ssAsH")
_MINsssAs = _make("MIN", "sssAs")
_MINsssdAs = _make("MIN", "sssdAs")
_MINsssssAs = _make("MIN", "sssssAs")
_MINsSeH = _make("MIN", "sSeH")
_MINdSe = _make("MIN", "dSe")
_MINssSe = _make("MIN", "ssSe")
_MINaaSe = _make("MIN", "aaSe")
_MINdssSe = _make("MIN", "dssSe")
_MINddssSe = _make("MIN", "ddssSe")
_MINsBr = _make("MIN", "sBr")
_MINsSnH3 = _make("MIN", "sSnH3")
_MINssSnH2 = _make("MIN", "ssSnH2")
_MINsssSnH = _make("MIN", "sssSnH")
_MINssssSn = _make("MIN", "ssssSn")
_MINsI = _make("MIN", "sI")
_MINsPbH3 = _make("MIN", "sPbH3")
_MINssPbH2 = _make("MIN", "ssPbH2")
_MINsssPbH = _make("MIN", "sssPbH")
_MINssssPb = _make("MIN", "ssssPb")


# ---- public wrappers (name == function_name; declared -> 0 bonus) ----


def NsLi(mol):
    """NsLi (E-State count of atom type sLi)."""
    return _NsLi(mol)


def NssBe(mol):
    """NssBe (E-State count of atom type ssBe)."""
    return _NssBe(mol)


def NssssBe(mol):
    """NssssBe (E-State count of atom type ssssBe)."""
    return _NssssBe(mol)


def NssBH(mol):
    """NssBH (E-State count of atom type ssBH)."""
    return _NssBH(mol)


def NsssB(mol):
    """NsssB (E-State count of atom type sssB)."""
    return _NsssB(mol)


def NssssB(mol):
    """NssssB (E-State count of atom type ssssB)."""
    return _NssssB(mol)


def NsCH3(mol):
    """NsCH3 (E-State count of atom type sCH3)."""
    return _NsCH3(mol)


def NdCH2(mol):
    """NdCH2 (E-State count of atom type dCH2)."""
    return _NdCH2(mol)


def NssCH2(mol):
    """NssCH2 (E-State count of atom type ssCH2)."""
    return _NssCH2(mol)


def NtCH(mol):
    """NtCH (E-State count of atom type tCH)."""
    return _NtCH(mol)


def NdsCH(mol):
    """NdsCH (E-State count of atom type dsCH)."""
    return _NdsCH(mol)


def NaaCH(mol):
    """NaaCH (E-State count of atom type aaCH)."""
    return _NaaCH(mol)


def NsssCH(mol):
    """NsssCH (E-State count of atom type sssCH)."""
    return _NsssCH(mol)


def NddC(mol):
    """NddC (E-State count of atom type ddC)."""
    return _NddC(mol)


def NtsC(mol):
    """NtsC (E-State count of atom type tsC)."""
    return _NtsC(mol)


def NdssC(mol):
    """NdssC (E-State count of atom type dssC)."""
    return _NdssC(mol)


def NaasC(mol):
    """NaasC (E-State count of atom type aasC)."""
    return _NaasC(mol)


def NaaaC(mol):
    """NaaaC (E-State count of atom type aaaC)."""
    return _NaaaC(mol)


def NssssC(mol):
    """NssssC (E-State count of atom type ssssC)."""
    return _NssssC(mol)


def NsNH3(mol):
    """NsNH3 (E-State count of atom type sNH3)."""
    return _NsNH3(mol)


def NsNH2(mol):
    """NsNH2 (E-State count of atom type sNH2)."""
    return _NsNH2(mol)


def NssNH2(mol):
    """NssNH2 (E-State count of atom type ssNH2)."""
    return _NssNH2(mol)


def NdNH(mol):
    """NdNH (E-State count of atom type dNH)."""
    return _NdNH(mol)


def NssNH(mol):
    """NssNH (E-State count of atom type ssNH)."""
    return _NssNH(mol)


def NaaNH(mol):
    """NaaNH (E-State count of atom type aaNH)."""
    return _NaaNH(mol)


def NtN(mol):
    """NtN (E-State count of atom type tN)."""
    return _NtN(mol)


def NsssNH(mol):
    """NsssNH (E-State count of atom type sssNH)."""
    return _NsssNH(mol)


def NdsN(mol):
    """NdsN (E-State count of atom type dsN)."""
    return _NdsN(mol)


def NaaN(mol):
    """NaaN (E-State count of atom type aaN)."""
    return _NaaN(mol)


def NsssN(mol):
    """NsssN (E-State count of atom type sssN)."""
    return _NsssN(mol)


def NddsN(mol):
    """NddsN (E-State count of atom type ddsN)."""
    return _NddsN(mol)


def NaasN(mol):
    """NaasN (E-State count of atom type aasN)."""
    return _NaasN(mol)


def NssssN(mol):
    """NssssN (E-State count of atom type ssssN)."""
    return _NssssN(mol)


def NsOH(mol):
    """NsOH (E-State count of atom type sOH)."""
    return _NsOH(mol)


def NdO(mol):
    """NdO (E-State count of atom type dO)."""
    return _NdO(mol)


def NssO(mol):
    """NssO (E-State count of atom type ssO)."""
    return _NssO(mol)


def NaaO(mol):
    """NaaO (E-State count of atom type aaO)."""
    return _NaaO(mol)


def NsF(mol):
    """NsF (E-State count of atom type sF)."""
    return _NsF(mol)


def NsSiH3(mol):
    """NsSiH3 (E-State count of atom type sSiH3)."""
    return _NsSiH3(mol)


def NssSiH2(mol):
    """NssSiH2 (E-State count of atom type ssSiH2)."""
    return _NssSiH2(mol)


def NsssSiH(mol):
    """NsssSiH (E-State count of atom type sssSiH)."""
    return _NsssSiH(mol)


def NssssSi(mol):
    """NssssSi (E-State count of atom type ssssSi)."""
    return _NssssSi(mol)


def NsPH2(mol):
    """NsPH2 (E-State count of atom type sPH2)."""
    return _NsPH2(mol)


def NssPH(mol):
    """NssPH (E-State count of atom type ssPH)."""
    return _NssPH(mol)


def NsssP(mol):
    """NsssP (E-State count of atom type sssP)."""
    return _NsssP(mol)


def NdsssP(mol):
    """NdsssP (E-State count of atom type dsssP)."""
    return _NdsssP(mol)


def NsssssP(mol):
    """NsssssP (E-State count of atom type sssssP)."""
    return _NsssssP(mol)


def NsSH(mol):
    """NsSH (E-State count of atom type sSH)."""
    return _NsSH(mol)


def NdS(mol):
    """NdS (E-State count of atom type dS)."""
    return _NdS(mol)


def NssS(mol):
    """NssS (E-State count of atom type ssS)."""
    return _NssS(mol)


def NaaS(mol):
    """NaaS (E-State count of atom type aaS)."""
    return _NaaS(mol)


def NdssS(mol):
    """NdssS (E-State count of atom type dssS)."""
    return _NdssS(mol)


def NddssS(mol):
    """NddssS (E-State count of atom type ddssS)."""
    return _NddssS(mol)


def NsCl(mol):
    """NsCl (E-State count of atom type sCl)."""
    return _NsCl(mol)


def NsGeH3(mol):
    """NsGeH3 (E-State count of atom type sGeH3)."""
    return _NsGeH3(mol)


def NssGeH2(mol):
    """NssGeH2 (E-State count of atom type ssGeH2)."""
    return _NssGeH2(mol)


def NsssGeH(mol):
    """NsssGeH (E-State count of atom type sssGeH)."""
    return _NsssGeH(mol)


def NssssGe(mol):
    """NssssGe (E-State count of atom type ssssGe)."""
    return _NssssGe(mol)


def NsAsH2(mol):
    """NsAsH2 (E-State count of atom type sAsH2)."""
    return _NsAsH2(mol)


def NssAsH(mol):
    """NssAsH (E-State count of atom type ssAsH)."""
    return _NssAsH(mol)


def NsssAs(mol):
    """NsssAs (E-State count of atom type sssAs)."""
    return _NsssAs(mol)


def NsssdAs(mol):
    """NsssdAs (E-State count of atom type sssdAs)."""
    return _NsssdAs(mol)


def NsssssAs(mol):
    """NsssssAs (E-State count of atom type sssssAs)."""
    return _NsssssAs(mol)


def NsSeH(mol):
    """NsSeH (E-State count of atom type sSeH)."""
    return _NsSeH(mol)


def NdSe(mol):
    """NdSe (E-State count of atom type dSe)."""
    return _NdSe(mol)


def NssSe(mol):
    """NssSe (E-State count of atom type ssSe)."""
    return _NssSe(mol)


def NaaSe(mol):
    """NaaSe (E-State count of atom type aaSe)."""
    return _NaaSe(mol)


def NdssSe(mol):
    """NdssSe (E-State count of atom type dssSe)."""
    return _NdssSe(mol)


def NddssSe(mol):
    """NddssSe (E-State count of atom type ddssSe)."""
    return _NddssSe(mol)


def NsBr(mol):
    """NsBr (E-State count of atom type sBr)."""
    return _NsBr(mol)


def NsSnH3(mol):
    """NsSnH3 (E-State count of atom type sSnH3)."""
    return _NsSnH3(mol)


def NssSnH2(mol):
    """NssSnH2 (E-State count of atom type ssSnH2)."""
    return _NssSnH2(mol)


def NsssSnH(mol):
    """NsssSnH (E-State count of atom type sssSnH)."""
    return _NsssSnH(mol)


def NssssSn(mol):
    """NssssSn (E-State count of atom type ssssSn)."""
    return _NssssSn(mol)


def NsI(mol):
    """NsI (E-State count of atom type sI)."""
    return _NsI(mol)


def NsPbH3(mol):
    """NsPbH3 (E-State count of atom type sPbH3)."""
    return _NsPbH3(mol)


def NssPbH2(mol):
    """NssPbH2 (E-State count of atom type ssPbH2)."""
    return _NssPbH2(mol)


def NsssPbH(mol):
    """NsssPbH (E-State count of atom type sssPbH)."""
    return _NsssPbH(mol)


def NssssPb(mol):
    """NssssPb (E-State count of atom type ssssPb)."""
    return _NssssPb(mol)


def SsLi(mol):
    """SsLi (E-State sum of atom type sLi)."""
    return _SsLi(mol)


def SssBe(mol):
    """SssBe (E-State sum of atom type ssBe)."""
    return _SssBe(mol)


def SssssBe(mol):
    """SssssBe (E-State sum of atom type ssssBe)."""
    return _SssssBe(mol)


def SssBH(mol):
    """SssBH (E-State sum of atom type ssBH)."""
    return _SssBH(mol)


def SsssB(mol):
    """SsssB (E-State sum of atom type sssB)."""
    return _SsssB(mol)


def SssssB(mol):
    """SssssB (E-State sum of atom type ssssB)."""
    return _SssssB(mol)


def SsCH3(mol):
    """SsCH3 (E-State sum of atom type sCH3)."""
    return _SsCH3(mol)


def SdCH2(mol):
    """SdCH2 (E-State sum of atom type dCH2)."""
    return _SdCH2(mol)


def SssCH2(mol):
    """SssCH2 (E-State sum of atom type ssCH2)."""
    return _SssCH2(mol)


def StCH(mol):
    """StCH (E-State sum of atom type tCH)."""
    return _StCH(mol)


def SdsCH(mol):
    """SdsCH (E-State sum of atom type dsCH)."""
    return _SdsCH(mol)


def SaaCH(mol):
    """SaaCH (E-State sum of atom type aaCH)."""
    return _SaaCH(mol)


def SsssCH(mol):
    """SsssCH (E-State sum of atom type sssCH)."""
    return _SsssCH(mol)


def SddC(mol):
    """SddC (E-State sum of atom type ddC)."""
    return _SddC(mol)


def StsC(mol):
    """StsC (E-State sum of atom type tsC)."""
    return _StsC(mol)


def SdssC(mol):
    """SdssC (E-State sum of atom type dssC)."""
    return _SdssC(mol)


def SaasC(mol):
    """SaasC (E-State sum of atom type aasC)."""
    return _SaasC(mol)


def SaaaC(mol):
    """SaaaC (E-State sum of atom type aaaC)."""
    return _SaaaC(mol)


def SssssC(mol):
    """SssssC (E-State sum of atom type ssssC)."""
    return _SssssC(mol)


def SsNH3(mol):
    """SsNH3 (E-State sum of atom type sNH3)."""
    return _SsNH3(mol)


def SsNH2(mol):
    """SsNH2 (E-State sum of atom type sNH2)."""
    return _SsNH2(mol)


def SssNH2(mol):
    """SssNH2 (E-State sum of atom type ssNH2)."""
    return _SssNH2(mol)


def SdNH(mol):
    """SdNH (E-State sum of atom type dNH)."""
    return _SdNH(mol)


def SssNH(mol):
    """SssNH (E-State sum of atom type ssNH)."""
    return _SssNH(mol)


def SaaNH(mol):
    """SaaNH (E-State sum of atom type aaNH)."""
    return _SaaNH(mol)


def StN(mol):
    """StN (E-State sum of atom type tN)."""
    return _StN(mol)


def SsssNH(mol):
    """SsssNH (E-State sum of atom type sssNH)."""
    return _SsssNH(mol)


def SdsN(mol):
    """SdsN (E-State sum of atom type dsN)."""
    return _SdsN(mol)


def SaaN(mol):
    """SaaN (E-State sum of atom type aaN)."""
    return _SaaN(mol)


def SsssN(mol):
    """SsssN (E-State sum of atom type sssN)."""
    return _SsssN(mol)


def SddsN(mol):
    """SddsN (E-State sum of atom type ddsN)."""
    return _SddsN(mol)


def SaasN(mol):
    """SaasN (E-State sum of atom type aasN)."""
    return _SaasN(mol)


def SssssN(mol):
    """SssssN (E-State sum of atom type ssssN)."""
    return _SssssN(mol)


def SsOH(mol):
    """SsOH (E-State sum of atom type sOH)."""
    return _SsOH(mol)


def SdO(mol):
    """SdO (E-State sum of atom type dO)."""
    return _SdO(mol)


def SssO(mol):
    """SssO (E-State sum of atom type ssO)."""
    return _SssO(mol)


def SaaO(mol):
    """SaaO (E-State sum of atom type aaO)."""
    return _SaaO(mol)


def SsF(mol):
    """SsF (E-State sum of atom type sF)."""
    return _SsF(mol)


def SsSiH3(mol):
    """SsSiH3 (E-State sum of atom type sSiH3)."""
    return _SsSiH3(mol)


def SssSiH2(mol):
    """SssSiH2 (E-State sum of atom type ssSiH2)."""
    return _SssSiH2(mol)


def SsssSiH(mol):
    """SsssSiH (E-State sum of atom type sssSiH)."""
    return _SsssSiH(mol)


def SssssSi(mol):
    """SssssSi (E-State sum of atom type ssssSi)."""
    return _SssssSi(mol)


def SsPH2(mol):
    """SsPH2 (E-State sum of atom type sPH2)."""
    return _SsPH2(mol)


def SssPH(mol):
    """SssPH (E-State sum of atom type ssPH)."""
    return _SssPH(mol)


def SsssP(mol):
    """SsssP (E-State sum of atom type sssP)."""
    return _SsssP(mol)


def SdsssP(mol):
    """SdsssP (E-State sum of atom type dsssP)."""
    return _SdsssP(mol)


def SsssssP(mol):
    """SsssssP (E-State sum of atom type sssssP)."""
    return _SsssssP(mol)


def SsSH(mol):
    """SsSH (E-State sum of atom type sSH)."""
    return _SsSH(mol)


def SdS(mol):
    """SdS (E-State sum of atom type dS)."""
    return _SdS(mol)


def SssS(mol):
    """SssS (E-State sum of atom type ssS)."""
    return _SssS(mol)


def SaaS(mol):
    """SaaS (E-State sum of atom type aaS)."""
    return _SaaS(mol)


def SdssS(mol):
    """SdssS (E-State sum of atom type dssS)."""
    return _SdssS(mol)


def SddssS(mol):
    """SddssS (E-State sum of atom type ddssS)."""
    return _SddssS(mol)


def SsCl(mol):
    """SsCl (E-State sum of atom type sCl)."""
    return _SsCl(mol)


def SsGeH3(mol):
    """SsGeH3 (E-State sum of atom type sGeH3)."""
    return _SsGeH3(mol)


def SssGeH2(mol):
    """SssGeH2 (E-State sum of atom type ssGeH2)."""
    return _SssGeH2(mol)


def SsssGeH(mol):
    """SsssGeH (E-State sum of atom type sssGeH)."""
    return _SsssGeH(mol)


def SssssGe(mol):
    """SssssGe (E-State sum of atom type ssssGe)."""
    return _SssssGe(mol)


def SsAsH2(mol):
    """SsAsH2 (E-State sum of atom type sAsH2)."""
    return _SsAsH2(mol)


def SssAsH(mol):
    """SssAsH (E-State sum of atom type ssAsH)."""
    return _SssAsH(mol)


def SsssAs(mol):
    """SsssAs (E-State sum of atom type sssAs)."""
    return _SsssAs(mol)


def SsssdAs(mol):
    """SsssdAs (E-State sum of atom type sssdAs)."""
    return _SsssdAs(mol)


def SsssssAs(mol):
    """SsssssAs (E-State sum of atom type sssssAs)."""
    return _SsssssAs(mol)


def SsSeH(mol):
    """SsSeH (E-State sum of atom type sSeH)."""
    return _SsSeH(mol)


def SdSe(mol):
    """SdSe (E-State sum of atom type dSe)."""
    return _SdSe(mol)


def SssSe(mol):
    """SssSe (E-State sum of atom type ssSe)."""
    return _SssSe(mol)


def SaaSe(mol):
    """SaaSe (E-State sum of atom type aaSe)."""
    return _SaaSe(mol)


def SdssSe(mol):
    """SdssSe (E-State sum of atom type dssSe)."""
    return _SdssSe(mol)


def SddssSe(mol):
    """SddssSe (E-State sum of atom type ddssSe)."""
    return _SddssSe(mol)


def SsBr(mol):
    """SsBr (E-State sum of atom type sBr)."""
    return _SsBr(mol)


def SsSnH3(mol):
    """SsSnH3 (E-State sum of atom type sSnH3)."""
    return _SsSnH3(mol)


def SssSnH2(mol):
    """SssSnH2 (E-State sum of atom type ssSnH2)."""
    return _SssSnH2(mol)


def SsssSnH(mol):
    """SsssSnH (E-State sum of atom type sssSnH)."""
    return _SsssSnH(mol)


def SssssSn(mol):
    """SssssSn (E-State sum of atom type ssssSn)."""
    return _SssssSn(mol)


def SsI(mol):
    """SsI (E-State sum of atom type sI)."""
    return _SsI(mol)


def SsPbH3(mol):
    """SsPbH3 (E-State sum of atom type sPbH3)."""
    return _SsPbH3(mol)


def SssPbH2(mol):
    """SssPbH2 (E-State sum of atom type ssPbH2)."""
    return _SssPbH2(mol)


def SsssPbH(mol):
    """SsssPbH (E-State sum of atom type sssPbH)."""
    return _SsssPbH(mol)


def SssssPb(mol):
    """SssssPb (E-State sum of atom type ssssPb)."""
    return _SssssPb(mol)


def MAXsLi(mol):
    """MAXsLi (E-State max of atom type sLi)."""
    return _MAXsLi(mol)


def MAXssBe(mol):
    """MAXssBe (E-State max of atom type ssBe)."""
    return _MAXssBe(mol)


def MAXssssBe(mol):
    """MAXssssBe (E-State max of atom type ssssBe)."""
    return _MAXssssBe(mol)


def MAXssBH(mol):
    """MAXssBH (E-State max of atom type ssBH)."""
    return _MAXssBH(mol)


def MAXsssB(mol):
    """MAXsssB (E-State max of atom type sssB)."""
    return _MAXsssB(mol)


def MAXssssB(mol):
    """MAXssssB (E-State max of atom type ssssB)."""
    return _MAXssssB(mol)


def MAXsCH3(mol):
    """MAXsCH3 (E-State max of atom type sCH3)."""
    return _MAXsCH3(mol)


def MAXdCH2(mol):
    """MAXdCH2 (E-State max of atom type dCH2)."""
    return _MAXdCH2(mol)


def MAXssCH2(mol):
    """MAXssCH2 (E-State max of atom type ssCH2)."""
    return _MAXssCH2(mol)


def MAXtCH(mol):
    """MAXtCH (E-State max of atom type tCH)."""
    return _MAXtCH(mol)


def MAXdsCH(mol):
    """MAXdsCH (E-State max of atom type dsCH)."""
    return _MAXdsCH(mol)


def MAXaaCH(mol):
    """MAXaaCH (E-State max of atom type aaCH)."""
    return _MAXaaCH(mol)


def MAXsssCH(mol):
    """MAXsssCH (E-State max of atom type sssCH)."""
    return _MAXsssCH(mol)


def MAXddC(mol):
    """MAXddC (E-State max of atom type ddC)."""
    return _MAXddC(mol)


def MAXtsC(mol):
    """MAXtsC (E-State max of atom type tsC)."""
    return _MAXtsC(mol)


def MAXdssC(mol):
    """MAXdssC (E-State max of atom type dssC)."""
    return _MAXdssC(mol)


def MAXaasC(mol):
    """MAXaasC (E-State max of atom type aasC)."""
    return _MAXaasC(mol)


def MAXaaaC(mol):
    """MAXaaaC (E-State max of atom type aaaC)."""
    return _MAXaaaC(mol)


def MAXssssC(mol):
    """MAXssssC (E-State max of atom type ssssC)."""
    return _MAXssssC(mol)


def MAXsNH3(mol):
    """MAXsNH3 (E-State max of atom type sNH3)."""
    return _MAXsNH3(mol)


def MAXsNH2(mol):
    """MAXsNH2 (E-State max of atom type sNH2)."""
    return _MAXsNH2(mol)


def MAXssNH2(mol):
    """MAXssNH2 (E-State max of atom type ssNH2)."""
    return _MAXssNH2(mol)


def MAXdNH(mol):
    """MAXdNH (E-State max of atom type dNH)."""
    return _MAXdNH(mol)


def MAXssNH(mol):
    """MAXssNH (E-State max of atom type ssNH)."""
    return _MAXssNH(mol)


def MAXaaNH(mol):
    """MAXaaNH (E-State max of atom type aaNH)."""
    return _MAXaaNH(mol)


def MAXtN(mol):
    """MAXtN (E-State max of atom type tN)."""
    return _MAXtN(mol)


def MAXsssNH(mol):
    """MAXsssNH (E-State max of atom type sssNH)."""
    return _MAXsssNH(mol)


def MAXdsN(mol):
    """MAXdsN (E-State max of atom type dsN)."""
    return _MAXdsN(mol)


def MAXaaN(mol):
    """MAXaaN (E-State max of atom type aaN)."""
    return _MAXaaN(mol)


def MAXsssN(mol):
    """MAXsssN (E-State max of atom type sssN)."""
    return _MAXsssN(mol)


def MAXddsN(mol):
    """MAXddsN (E-State max of atom type ddsN)."""
    return _MAXddsN(mol)


def MAXaasN(mol):
    """MAXaasN (E-State max of atom type aasN)."""
    return _MAXaasN(mol)


def MAXssssN(mol):
    """MAXssssN (E-State max of atom type ssssN)."""
    return _MAXssssN(mol)


def MAXsOH(mol):
    """MAXsOH (E-State max of atom type sOH)."""
    return _MAXsOH(mol)


def MAXdO(mol):
    """MAXdO (E-State max of atom type dO)."""
    return _MAXdO(mol)


def MAXssO(mol):
    """MAXssO (E-State max of atom type ssO)."""
    return _MAXssO(mol)


def MAXaaO(mol):
    """MAXaaO (E-State max of atom type aaO)."""
    return _MAXaaO(mol)


def MAXsF(mol):
    """MAXsF (E-State max of atom type sF)."""
    return _MAXsF(mol)


def MAXsSiH3(mol):
    """MAXsSiH3 (E-State max of atom type sSiH3)."""
    return _MAXsSiH3(mol)


def MAXssSiH2(mol):
    """MAXssSiH2 (E-State max of atom type ssSiH2)."""
    return _MAXssSiH2(mol)


def MAXsssSiH(mol):
    """MAXsssSiH (E-State max of atom type sssSiH)."""
    return _MAXsssSiH(mol)


def MAXssssSi(mol):
    """MAXssssSi (E-State max of atom type ssssSi)."""
    return _MAXssssSi(mol)


def MAXsPH2(mol):
    """MAXsPH2 (E-State max of atom type sPH2)."""
    return _MAXsPH2(mol)


def MAXssPH(mol):
    """MAXssPH (E-State max of atom type ssPH)."""
    return _MAXssPH(mol)


def MAXsssP(mol):
    """MAXsssP (E-State max of atom type sssP)."""
    return _MAXsssP(mol)


def MAXdsssP(mol):
    """MAXdsssP (E-State max of atom type dsssP)."""
    return _MAXdsssP(mol)


def MAXsssssP(mol):
    """MAXsssssP (E-State max of atom type sssssP)."""
    return _MAXsssssP(mol)


def MAXsSH(mol):
    """MAXsSH (E-State max of atom type sSH)."""
    return _MAXsSH(mol)


def MAXdS(mol):
    """MAXdS (E-State max of atom type dS)."""
    return _MAXdS(mol)


def MAXssS(mol):
    """MAXssS (E-State max of atom type ssS)."""
    return _MAXssS(mol)


def MAXaaS(mol):
    """MAXaaS (E-State max of atom type aaS)."""
    return _MAXaaS(mol)


def MAXdssS(mol):
    """MAXdssS (E-State max of atom type dssS)."""
    return _MAXdssS(mol)


def MAXddssS(mol):
    """MAXddssS (E-State max of atom type ddssS)."""
    return _MAXddssS(mol)


def MAXsCl(mol):
    """MAXsCl (E-State max of atom type sCl)."""
    return _MAXsCl(mol)


def MAXsGeH3(mol):
    """MAXsGeH3 (E-State max of atom type sGeH3)."""
    return _MAXsGeH3(mol)


def MAXssGeH2(mol):
    """MAXssGeH2 (E-State max of atom type ssGeH2)."""
    return _MAXssGeH2(mol)


def MAXsssGeH(mol):
    """MAXsssGeH (E-State max of atom type sssGeH)."""
    return _MAXsssGeH(mol)


def MAXssssGe(mol):
    """MAXssssGe (E-State max of atom type ssssGe)."""
    return _MAXssssGe(mol)


def MAXsAsH2(mol):
    """MAXsAsH2 (E-State max of atom type sAsH2)."""
    return _MAXsAsH2(mol)


def MAXssAsH(mol):
    """MAXssAsH (E-State max of atom type ssAsH)."""
    return _MAXssAsH(mol)


def MAXsssAs(mol):
    """MAXsssAs (E-State max of atom type sssAs)."""
    return _MAXsssAs(mol)


def MAXsssdAs(mol):
    """MAXsssdAs (E-State max of atom type sssdAs)."""
    return _MAXsssdAs(mol)


def MAXsssssAs(mol):
    """MAXsssssAs (E-State max of atom type sssssAs)."""
    return _MAXsssssAs(mol)


def MAXsSeH(mol):
    """MAXsSeH (E-State max of atom type sSeH)."""
    return _MAXsSeH(mol)


def MAXdSe(mol):
    """MAXdSe (E-State max of atom type dSe)."""
    return _MAXdSe(mol)


def MAXssSe(mol):
    """MAXssSe (E-State max of atom type ssSe)."""
    return _MAXssSe(mol)


def MAXaaSe(mol):
    """MAXaaSe (E-State max of atom type aaSe)."""
    return _MAXaaSe(mol)


def MAXdssSe(mol):
    """MAXdssSe (E-State max of atom type dssSe)."""
    return _MAXdssSe(mol)


def MAXddssSe(mol):
    """MAXddssSe (E-State max of atom type ddssSe)."""
    return _MAXddssSe(mol)


def MAXsBr(mol):
    """MAXsBr (E-State max of atom type sBr)."""
    return _MAXsBr(mol)


def MAXsSnH3(mol):
    """MAXsSnH3 (E-State max of atom type sSnH3)."""
    return _MAXsSnH3(mol)


def MAXssSnH2(mol):
    """MAXssSnH2 (E-State max of atom type ssSnH2)."""
    return _MAXssSnH2(mol)


def MAXsssSnH(mol):
    """MAXsssSnH (E-State max of atom type sssSnH)."""
    return _MAXsssSnH(mol)


def MAXssssSn(mol):
    """MAXssssSn (E-State max of atom type ssssSn)."""
    return _MAXssssSn(mol)


def MAXsI(mol):
    """MAXsI (E-State max of atom type sI)."""
    return _MAXsI(mol)


def MAXsPbH3(mol):
    """MAXsPbH3 (E-State max of atom type sPbH3)."""
    return _MAXsPbH3(mol)


def MAXssPbH2(mol):
    """MAXssPbH2 (E-State max of atom type ssPbH2)."""
    return _MAXssPbH2(mol)


def MAXsssPbH(mol):
    """MAXsssPbH (E-State max of atom type sssPbH)."""
    return _MAXsssPbH(mol)


def MAXssssPb(mol):
    """MAXssssPb (E-State max of atom type ssssPb)."""
    return _MAXssssPb(mol)


def MINsLi(mol):
    """MINsLi (E-State min of atom type sLi)."""
    return _MINsLi(mol)


def MINssBe(mol):
    """MINssBe (E-State min of atom type ssBe)."""
    return _MINssBe(mol)


def MINssssBe(mol):
    """MINssssBe (E-State min of atom type ssssBe)."""
    return _MINssssBe(mol)


def MINssBH(mol):
    """MINssBH (E-State min of atom type ssBH)."""
    return _MINssBH(mol)


def MINsssB(mol):
    """MINsssB (E-State min of atom type sssB)."""
    return _MINsssB(mol)


def MINssssB(mol):
    """MINssssB (E-State min of atom type ssssB)."""
    return _MINssssB(mol)


def MINsCH3(mol):
    """MINsCH3 (E-State min of atom type sCH3)."""
    return _MINsCH3(mol)


def MINdCH2(mol):
    """MINdCH2 (E-State min of atom type dCH2)."""
    return _MINdCH2(mol)


def MINssCH2(mol):
    """MINssCH2 (E-State min of atom type ssCH2)."""
    return _MINssCH2(mol)


def MINtCH(mol):
    """MINtCH (E-State min of atom type tCH)."""
    return _MINtCH(mol)


def MINdsCH(mol):
    """MINdsCH (E-State min of atom type dsCH)."""
    return _MINdsCH(mol)


def MINaaCH(mol):
    """MINaaCH (E-State min of atom type aaCH)."""
    return _MINaaCH(mol)


def MINsssCH(mol):
    """MINsssCH (E-State min of atom type sssCH)."""
    return _MINsssCH(mol)


def MINddC(mol):
    """MINddC (E-State min of atom type ddC)."""
    return _MINddC(mol)


def MINtsC(mol):
    """MINtsC (E-State min of atom type tsC)."""
    return _MINtsC(mol)


def MINdssC(mol):
    """MINdssC (E-State min of atom type dssC)."""
    return _MINdssC(mol)


def MINaasC(mol):
    """MINaasC (E-State min of atom type aasC)."""
    return _MINaasC(mol)


def MINaaaC(mol):
    """MINaaaC (E-State min of atom type aaaC)."""
    return _MINaaaC(mol)


def MINssssC(mol):
    """MINssssC (E-State min of atom type ssssC)."""
    return _MINssssC(mol)


def MINsNH3(mol):
    """MINsNH3 (E-State min of atom type sNH3)."""
    return _MINsNH3(mol)


def MINsNH2(mol):
    """MINsNH2 (E-State min of atom type sNH2)."""
    return _MINsNH2(mol)


def MINssNH2(mol):
    """MINssNH2 (E-State min of atom type ssNH2)."""
    return _MINssNH2(mol)


def MINdNH(mol):
    """MINdNH (E-State min of atom type dNH)."""
    return _MINdNH(mol)


def MINssNH(mol):
    """MINssNH (E-State min of atom type ssNH)."""
    return _MINssNH(mol)


def MINaaNH(mol):
    """MINaaNH (E-State min of atom type aaNH)."""
    return _MINaaNH(mol)


def MINtN(mol):
    """MINtN (E-State min of atom type tN)."""
    return _MINtN(mol)


def MINsssNH(mol):
    """MINsssNH (E-State min of atom type sssNH)."""
    return _MINsssNH(mol)


def MINdsN(mol):
    """MINdsN (E-State min of atom type dsN)."""
    return _MINdsN(mol)


def MINaaN(mol):
    """MINaaN (E-State min of atom type aaN)."""
    return _MINaaN(mol)


def MINsssN(mol):
    """MINsssN (E-State min of atom type sssN)."""
    return _MINsssN(mol)


def MINddsN(mol):
    """MINddsN (E-State min of atom type ddsN)."""
    return _MINddsN(mol)


def MINaasN(mol):
    """MINaasN (E-State min of atom type aasN)."""
    return _MINaasN(mol)


def MINssssN(mol):
    """MINssssN (E-State min of atom type ssssN)."""
    return _MINssssN(mol)


def MINsOH(mol):
    """MINsOH (E-State min of atom type sOH)."""
    return _MINsOH(mol)


def MINdO(mol):
    """MINdO (E-State min of atom type dO)."""
    return _MINdO(mol)


def MINssO(mol):
    """MINssO (E-State min of atom type ssO)."""
    return _MINssO(mol)


def MINaaO(mol):
    """MINaaO (E-State min of atom type aaO)."""
    return _MINaaO(mol)


def MINsF(mol):
    """MINsF (E-State min of atom type sF)."""
    return _MINsF(mol)


def MINsSiH3(mol):
    """MINsSiH3 (E-State min of atom type sSiH3)."""
    return _MINsSiH3(mol)


def MINssSiH2(mol):
    """MINssSiH2 (E-State min of atom type ssSiH2)."""
    return _MINssSiH2(mol)


def MINsssSiH(mol):
    """MINsssSiH (E-State min of atom type sssSiH)."""
    return _MINsssSiH(mol)


def MINssssSi(mol):
    """MINssssSi (E-State min of atom type ssssSi)."""
    return _MINssssSi(mol)


def MINsPH2(mol):
    """MINsPH2 (E-State min of atom type sPH2)."""
    return _MINsPH2(mol)


def MINssPH(mol):
    """MINssPH (E-State min of atom type ssPH)."""
    return _MINssPH(mol)


def MINsssP(mol):
    """MINsssP (E-State min of atom type sssP)."""
    return _MINsssP(mol)


def MINdsssP(mol):
    """MINdsssP (E-State min of atom type dsssP)."""
    return _MINdsssP(mol)


def MINsssssP(mol):
    """MINsssssP (E-State min of atom type sssssP)."""
    return _MINsssssP(mol)


def MINsSH(mol):
    """MINsSH (E-State min of atom type sSH)."""
    return _MINsSH(mol)


def MINdS(mol):
    """MINdS (E-State min of atom type dS)."""
    return _MINdS(mol)


def MINssS(mol):
    """MINssS (E-State min of atom type ssS)."""
    return _MINssS(mol)


def MINaaS(mol):
    """MINaaS (E-State min of atom type aaS)."""
    return _MINaaS(mol)


def MINdssS(mol):
    """MINdssS (E-State min of atom type dssS)."""
    return _MINdssS(mol)


def MINddssS(mol):
    """MINddssS (E-State min of atom type ddssS)."""
    return _MINddssS(mol)


def MINsCl(mol):
    """MINsCl (E-State min of atom type sCl)."""
    return _MINsCl(mol)


def MINsGeH3(mol):
    """MINsGeH3 (E-State min of atom type sGeH3)."""
    return _MINsGeH3(mol)


def MINssGeH2(mol):
    """MINssGeH2 (E-State min of atom type ssGeH2)."""
    return _MINssGeH2(mol)


def MINsssGeH(mol):
    """MINsssGeH (E-State min of atom type sssGeH)."""
    return _MINsssGeH(mol)


def MINssssGe(mol):
    """MINssssGe (E-State min of atom type ssssGe)."""
    return _MINssssGe(mol)


def MINsAsH2(mol):
    """MINsAsH2 (E-State min of atom type sAsH2)."""
    return _MINsAsH2(mol)


def MINssAsH(mol):
    """MINssAsH (E-State min of atom type ssAsH)."""
    return _MINssAsH(mol)


def MINsssAs(mol):
    """MINsssAs (E-State min of atom type sssAs)."""
    return _MINsssAs(mol)


def MINsssdAs(mol):
    """MINsssdAs (E-State min of atom type sssdAs)."""
    return _MINsssdAs(mol)


def MINsssssAs(mol):
    """MINsssssAs (E-State min of atom type sssssAs)."""
    return _MINsssssAs(mol)


def MINsSeH(mol):
    """MINsSeH (E-State min of atom type sSeH)."""
    return _MINsSeH(mol)


def MINdSe(mol):
    """MINdSe (E-State min of atom type dSe)."""
    return _MINdSe(mol)


def MINssSe(mol):
    """MINssSe (E-State min of atom type ssSe)."""
    return _MINssSe(mol)


def MINaaSe(mol):
    """MINaaSe (E-State min of atom type aaSe)."""
    return _MINaaSe(mol)


def MINdssSe(mol):
    """MINdssSe (E-State min of atom type dssSe)."""
    return _MINdssSe(mol)


def MINddssSe(mol):
    """MINddssSe (E-State min of atom type ddssSe)."""
    return _MINddssSe(mol)


def MINsBr(mol):
    """MINsBr (E-State min of atom type sBr)."""
    return _MINsBr(mol)


def MINsSnH3(mol):
    """MINsSnH3 (E-State min of atom type sSnH3)."""
    return _MINsSnH3(mol)


def MINssSnH2(mol):
    """MINssSnH2 (E-State min of atom type ssSnH2)."""
    return _MINssSnH2(mol)


def MINsssSnH(mol):
    """MINsssSnH (E-State min of atom type sssSnH)."""
    return _MINsssSnH(mol)


def MINssssSn(mol):
    """MINssssSn (E-State min of atom type ssssSn)."""
    return _MINssssSn(mol)


def MINsI(mol):
    """MINsI (E-State min of atom type sI)."""
    return _MINsI(mol)


def MINsPbH3(mol):
    """MINsPbH3 (E-State min of atom type sPbH3)."""
    return _MINsPbH3(mol)


def MINssPbH2(mol):
    """MINssPbH2 (E-State min of atom type ssPbH2)."""
    return _MINssPbH2(mol)


def MINsssPbH(mol):
    """MINsssPbH (E-State min of atom type sssPbH)."""
    return _MINsssPbH(mol)


def MINssssPb(mol):
    """MINssssPb (E-State min of atom type ssssPb)."""
    return _MINssssPb(mol)


ALL_DESCRIPTOR_NAMES = (
    "NsLi",
    "NssBe",
    "NssssBe",
    "NssBH",
    "NsssB",
    "NssssB",
    "NsCH3",
    "NdCH2",
    "NssCH2",
    "NtCH",
    "NdsCH",
    "NaaCH",
    "NsssCH",
    "NddC",
    "NtsC",
    "NdssC",
    "NaasC",
    "NaaaC",
    "NssssC",
    "NsNH3",
    "NsNH2",
    "NssNH2",
    "NdNH",
    "NssNH",
    "NaaNH",
    "NtN",
    "NsssNH",
    "NdsN",
    "NaaN",
    "NsssN",
    "NddsN",
    "NaasN",
    "NssssN",
    "NsOH",
    "NdO",
    "NssO",
    "NaaO",
    "NsF",
    "NsSiH3",
    "NssSiH2",
    "NsssSiH",
    "NssssSi",
    "NsPH2",
    "NssPH",
    "NsssP",
    "NdsssP",
    "NsssssP",
    "NsSH",
    "NdS",
    "NssS",
    "NaaS",
    "NdssS",
    "NddssS",
    "NsCl",
    "NsGeH3",
    "NssGeH2",
    "NsssGeH",
    "NssssGe",
    "NsAsH2",
    "NssAsH",
    "NsssAs",
    "NsssdAs",
    "NsssssAs",
    "NsSeH",
    "NdSe",
    "NssSe",
    "NaaSe",
    "NdssSe",
    "NddssSe",
    "NsBr",
    "NsSnH3",
    "NssSnH2",
    "NsssSnH",
    "NssssSn",
    "NsI",
    "NsPbH3",
    "NssPbH2",
    "NsssPbH",
    "NssssPb",
    "SsLi",
    "SssBe",
    "SssssBe",
    "SssBH",
    "SsssB",
    "SssssB",
    "SsCH3",
    "SdCH2",
    "SssCH2",
    "StCH",
    "SdsCH",
    "SaaCH",
    "SsssCH",
    "SddC",
    "StsC",
    "SdssC",
    "SaasC",
    "SaaaC",
    "SssssC",
    "SsNH3",
    "SsNH2",
    "SssNH2",
    "SdNH",
    "SssNH",
    "SaaNH",
    "StN",
    "SsssNH",
    "SdsN",
    "SaaN",
    "SsssN",
    "SddsN",
    "SaasN",
    "SssssN",
    "SsOH",
    "SdO",
    "SssO",
    "SaaO",
    "SsF",
    "SsSiH3",
    "SssSiH2",
    "SsssSiH",
    "SssssSi",
    "SsPH2",
    "SssPH",
    "SsssP",
    "SdsssP",
    "SsssssP",
    "SsSH",
    "SdS",
    "SssS",
    "SaaS",
    "SdssS",
    "SddssS",
    "SsCl",
    "SsGeH3",
    "SssGeH2",
    "SsssGeH",
    "SssssGe",
    "SsAsH2",
    "SssAsH",
    "SsssAs",
    "SsssdAs",
    "SsssssAs",
    "SsSeH",
    "SdSe",
    "SssSe",
    "SaaSe",
    "SdssSe",
    "SddssSe",
    "SsBr",
    "SsSnH3",
    "SssSnH2",
    "SsssSnH",
    "SssssSn",
    "SsI",
    "SsPbH3",
    "SssPbH2",
    "SsssPbH",
    "SssssPb",
    "MAXsLi",
    "MAXssBe",
    "MAXssssBe",
    "MAXssBH",
    "MAXsssB",
    "MAXssssB",
    "MAXsCH3",
    "MAXdCH2",
    "MAXssCH2",
    "MAXtCH",
    "MAXdsCH",
    "MAXaaCH",
    "MAXsssCH",
    "MAXddC",
    "MAXtsC",
    "MAXdssC",
    "MAXaasC",
    "MAXaaaC",
    "MAXssssC",
    "MAXsNH3",
    "MAXsNH2",
    "MAXssNH2",
    "MAXdNH",
    "MAXssNH",
    "MAXaaNH",
    "MAXtN",
    "MAXsssNH",
    "MAXdsN",
    "MAXaaN",
    "MAXsssN",
    "MAXddsN",
    "MAXaasN",
    "MAXssssN",
    "MAXsOH",
    "MAXdO",
    "MAXssO",
    "MAXaaO",
    "MAXsF",
    "MAXsSiH3",
    "MAXssSiH2",
    "MAXsssSiH",
    "MAXssssSi",
    "MAXsPH2",
    "MAXssPH",
    "MAXsssP",
    "MAXdsssP",
    "MAXsssssP",
    "MAXsSH",
    "MAXdS",
    "MAXssS",
    "MAXaaS",
    "MAXdssS",
    "MAXddssS",
    "MAXsCl",
    "MAXsGeH3",
    "MAXssGeH2",
    "MAXsssGeH",
    "MAXssssGe",
    "MAXsAsH2",
    "MAXssAsH",
    "MAXsssAs",
    "MAXsssdAs",
    "MAXsssssAs",
    "MAXsSeH",
    "MAXdSe",
    "MAXssSe",
    "MAXaaSe",
    "MAXdssSe",
    "MAXddssSe",
    "MAXsBr",
    "MAXsSnH3",
    "MAXssSnH2",
    "MAXsssSnH",
    "MAXssssSn",
    "MAXsI",
    "MAXsPbH3",
    "MAXssPbH2",
    "MAXsssPbH",
    "MAXssssPb",
    "MINsLi",
    "MINssBe",
    "MINssssBe",
    "MINssBH",
    "MINsssB",
    "MINssssB",
    "MINsCH3",
    "MINdCH2",
    "MINssCH2",
    "MINtCH",
    "MINdsCH",
    "MINaaCH",
    "MINsssCH",
    "MINddC",
    "MINtsC",
    "MINdssC",
    "MINaasC",
    "MINaaaC",
    "MINssssC",
    "MINsNH3",
    "MINsNH2",
    "MINssNH2",
    "MINdNH",
    "MINssNH",
    "MINaaNH",
    "MINtN",
    "MINsssNH",
    "MINdsN",
    "MINaaN",
    "MINsssN",
    "MINddsN",
    "MINaasN",
    "MINssssN",
    "MINsOH",
    "MINdO",
    "MINssO",
    "MINaaO",
    "MINsF",
    "MINsSiH3",
    "MINssSiH2",
    "MINsssSiH",
    "MINssssSi",
    "MINsPH2",
    "MINssPH",
    "MINsssP",
    "MINdsssP",
    "MINsssssP",
    "MINsSH",
    "MINdS",
    "MINssS",
    "MINaaS",
    "MINdssS",
    "MINddssS",
    "MINsCl",
    "MINsGeH3",
    "MINssGeH2",
    "MINsssGeH",
    "MINssssGe",
    "MINsAsH2",
    "MINssAsH",
    "MINsssAs",
    "MINsssdAs",
    "MINsssssAs",
    "MINsSeH",
    "MINdSe",
    "MINssSe",
    "MINaaSe",
    "MINdssSe",
    "MINddssSe",
    "MINsBr",
    "MINsSnH3",
    "MINssSnH2",
    "MINsssSnH",
    "MINssssSn",
    "MINsI",
    "MINsPbH3",
    "MINssPbH2",
    "MINsssPbH",
    "MINssssPb",
)
