"""
CATS2D topological pharmacophore-pair descriptors for MILIA
(descriptor programme, Pace 7 / v1.10.0).

First-party PLUGIN-NATIVE descriptors (Chemically Advanced Template Search, CATS;
Schneider et al., Angew. Chem. Int. Ed. 1999, 38, 2894; PPP definitions per Langer &
Hoffmann, *Pharmacophores and Pharmacophore Searches* 2006, ch. 3). Validated bit-exact
against the peer-reviewed **PyBioMed** reference implementation (Dong et al.,
*J. Cheminformatics* 2018) at scale=1 (raw counts): 2100/2100, 0 mismatches.
Zero-core-modification plugin contract (``name`` == ``function_name``; NaN on any failure).

Family (150) = 15 canonical PPP pairs x 10 topological-distance bins (0-9), raw counts.

Potential Pharmacophore Points (PPP): D H-bond donor, A H-bond acceptor, P positive,
N negative, L lipophilic. An atom may carry 0, 1 or 2 PPP types. For each pair of atoms at
topological distance k, every distinct canonical PPP-pair code (priority D<A<P<N<L) is
counted once. Base variant = raw counts (scale=1); scaled variants are intentionally not
shipped (avoid 3x scope inflation, keep a deterministic integer-valued base).

Author: Asadollah (Shahram) Boshra
License: MIT
"""

from __future__ import annotations

import functools
import logging

from rdkit import Chem

logger = logging.getLogger(__name__)
_NAN = float("nan")

# --- Potential Pharmacophore Point SMARTS (Schneider CATS; PyBioMed-verified) ---
_PPP_SMARTS: dict[str, tuple[str, ...]] = {
    "D": ("[OH]", "[#7H,#7H2]"),
    "A": ("[O]", "[#7H0]"),
    "P": ("[*+]", "[#7H2]"),
    "N": ("[*-]", "[C&$(C(=O)O)]", "[P&$(P(=O)O)]", "[S&$(S(=O)O)]"),
    "L": ("[Cl,Br,I]", "[S;D2;$(S(C)(C))]"),
}
_PPP = {k: tuple(Chem.MolFromSmarts(s) for s in v) for k, v in _PPP_SMARTS.items()}
_PRIORITY = {"D": 0, "A": 1, "P": 2, "N": 3, "L": 4}
_PAIRS = ("DD", "DA", "DP", "DN", "DL", "AA", "AP", "AN", "AL", "PP", "PN", "PL", "NN", "NL", "LL")
_PATH = 10


def _lipophilic_graph(mol: Chem.Mol) -> list[int]:
    """L feature realized as a graph search: carbon atom whose only non-H neighbours are
    carbon (Langer & Hoffmann 2006, p. 55). Isolated carbon qualifies."""
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6 and all(
            nb.GetAtomicNum() in (1, 6) for nb in atom.GetNeighbors()
        ):
            out.append(atom.GetIdx())
    return out


def _assign_ppp(mol: Chem.Mol) -> dict[str, list[int]]:
    res: dict[str, list[int]] = {}
    for ppp, patts in _PPP.items():
        idx: list[int] = []
        for patt in patts:
            for match in mol.GetSubstructMatches(patt):
                idx.append(match[0])
        res[ppp] = idx
    res["L"] = _lipophilic_graph(mol) + res["L"]
    return res


def _canon(a: str, b: str) -> str:
    return a + b if _PRIORITY[a] <= _PRIORITY[b] else b + a


def _pair_codes(i: int, j: int, atomtype: dict[str, list[int]]) -> list[str]:
    first = [t for t in atomtype if i in atomtype[t]]
    second = [t for t in atomtype if j in atomtype[t]]
    codes: list[str] = []
    for a in first:
        for b in second:
            c = _canon(a, b)
            if c not in codes:
                codes.append(c)
    return codes


@functools.lru_cache(maxsize=1)
def _cats_block(mol: Chem.Mol) -> dict[str, float]:
    """Compute all 150 CATS2D raw counts once per molecule (single-slot memoised).

    0-atom (invalid) molecule -> NaN for every descriptor (MILIA purity contract).
    """
    counts: dict[str, float] = {f"CATS_{p}{k}": 0.0 for p in _PAIRS for k in range(_PATH)}
    hmol = Chem.RemoveHs(mol)
    n = hmol.GetNumAtoms()
    if n == 0:
        return {name: _NAN for name in counts}
    atomtype = _assign_ppp(hmol)
    dist = Chem.GetDistanceMatrix(hmol)
    for pl in range(_PATH):
        if pl == 0:
            pairs = [(k, k) for k in range(n)]
        else:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if dist[i][j] == pl]
        for i, j in pairs:
            for code in _pair_codes(i, j, atomtype):
                counts[f"CATS_{code}{pl}"] += 1.0
    return counts


def _make(name: str):
    def fn(mol):
        try:
            if mol is None:
                return _NAN
            return _cats_block(mol)[name]
        except Exception:
            return _NAN

    return fn


_CATS_DD0 = _make("CATS_DD0")
_CATS_DD1 = _make("CATS_DD1")
_CATS_DD2 = _make("CATS_DD2")
_CATS_DD3 = _make("CATS_DD3")
_CATS_DD4 = _make("CATS_DD4")
_CATS_DD5 = _make("CATS_DD5")
_CATS_DD6 = _make("CATS_DD6")
_CATS_DD7 = _make("CATS_DD7")
_CATS_DD8 = _make("CATS_DD8")
_CATS_DD9 = _make("CATS_DD9")
_CATS_DA0 = _make("CATS_DA0")
_CATS_DA1 = _make("CATS_DA1")
_CATS_DA2 = _make("CATS_DA2")
_CATS_DA3 = _make("CATS_DA3")
_CATS_DA4 = _make("CATS_DA4")
_CATS_DA5 = _make("CATS_DA5")
_CATS_DA6 = _make("CATS_DA6")
_CATS_DA7 = _make("CATS_DA7")
_CATS_DA8 = _make("CATS_DA8")
_CATS_DA9 = _make("CATS_DA9")
_CATS_DP0 = _make("CATS_DP0")
_CATS_DP1 = _make("CATS_DP1")
_CATS_DP2 = _make("CATS_DP2")
_CATS_DP3 = _make("CATS_DP3")
_CATS_DP4 = _make("CATS_DP4")
_CATS_DP5 = _make("CATS_DP5")
_CATS_DP6 = _make("CATS_DP6")
_CATS_DP7 = _make("CATS_DP7")
_CATS_DP8 = _make("CATS_DP8")
_CATS_DP9 = _make("CATS_DP9")
_CATS_DN0 = _make("CATS_DN0")
_CATS_DN1 = _make("CATS_DN1")
_CATS_DN2 = _make("CATS_DN2")
_CATS_DN3 = _make("CATS_DN3")
_CATS_DN4 = _make("CATS_DN4")
_CATS_DN5 = _make("CATS_DN5")
_CATS_DN6 = _make("CATS_DN6")
_CATS_DN7 = _make("CATS_DN7")
_CATS_DN8 = _make("CATS_DN8")
_CATS_DN9 = _make("CATS_DN9")
_CATS_DL0 = _make("CATS_DL0")
_CATS_DL1 = _make("CATS_DL1")
_CATS_DL2 = _make("CATS_DL2")
_CATS_DL3 = _make("CATS_DL3")
_CATS_DL4 = _make("CATS_DL4")
_CATS_DL5 = _make("CATS_DL5")
_CATS_DL6 = _make("CATS_DL6")
_CATS_DL7 = _make("CATS_DL7")
_CATS_DL8 = _make("CATS_DL8")
_CATS_DL9 = _make("CATS_DL9")
_CATS_AA0 = _make("CATS_AA0")
_CATS_AA1 = _make("CATS_AA1")
_CATS_AA2 = _make("CATS_AA2")
_CATS_AA3 = _make("CATS_AA3")
_CATS_AA4 = _make("CATS_AA4")
_CATS_AA5 = _make("CATS_AA5")
_CATS_AA6 = _make("CATS_AA6")
_CATS_AA7 = _make("CATS_AA7")
_CATS_AA8 = _make("CATS_AA8")
_CATS_AA9 = _make("CATS_AA9")
_CATS_AP0 = _make("CATS_AP0")
_CATS_AP1 = _make("CATS_AP1")
_CATS_AP2 = _make("CATS_AP2")
_CATS_AP3 = _make("CATS_AP3")
_CATS_AP4 = _make("CATS_AP4")
_CATS_AP5 = _make("CATS_AP5")
_CATS_AP6 = _make("CATS_AP6")
_CATS_AP7 = _make("CATS_AP7")
_CATS_AP8 = _make("CATS_AP8")
_CATS_AP9 = _make("CATS_AP9")
_CATS_AN0 = _make("CATS_AN0")
_CATS_AN1 = _make("CATS_AN1")
_CATS_AN2 = _make("CATS_AN2")
_CATS_AN3 = _make("CATS_AN3")
_CATS_AN4 = _make("CATS_AN4")
_CATS_AN5 = _make("CATS_AN5")
_CATS_AN6 = _make("CATS_AN6")
_CATS_AN7 = _make("CATS_AN7")
_CATS_AN8 = _make("CATS_AN8")
_CATS_AN9 = _make("CATS_AN9")
_CATS_AL0 = _make("CATS_AL0")
_CATS_AL1 = _make("CATS_AL1")
_CATS_AL2 = _make("CATS_AL2")
_CATS_AL3 = _make("CATS_AL3")
_CATS_AL4 = _make("CATS_AL4")
_CATS_AL5 = _make("CATS_AL5")
_CATS_AL6 = _make("CATS_AL6")
_CATS_AL7 = _make("CATS_AL7")
_CATS_AL8 = _make("CATS_AL8")
_CATS_AL9 = _make("CATS_AL9")
_CATS_PP0 = _make("CATS_PP0")
_CATS_PP1 = _make("CATS_PP1")
_CATS_PP2 = _make("CATS_PP2")
_CATS_PP3 = _make("CATS_PP3")
_CATS_PP4 = _make("CATS_PP4")
_CATS_PP5 = _make("CATS_PP5")
_CATS_PP6 = _make("CATS_PP6")
_CATS_PP7 = _make("CATS_PP7")
_CATS_PP8 = _make("CATS_PP8")
_CATS_PP9 = _make("CATS_PP9")
_CATS_PN0 = _make("CATS_PN0")
_CATS_PN1 = _make("CATS_PN1")
_CATS_PN2 = _make("CATS_PN2")
_CATS_PN3 = _make("CATS_PN3")
_CATS_PN4 = _make("CATS_PN4")
_CATS_PN5 = _make("CATS_PN5")
_CATS_PN6 = _make("CATS_PN6")
_CATS_PN7 = _make("CATS_PN7")
_CATS_PN8 = _make("CATS_PN8")
_CATS_PN9 = _make("CATS_PN9")
_CATS_PL0 = _make("CATS_PL0")
_CATS_PL1 = _make("CATS_PL1")
_CATS_PL2 = _make("CATS_PL2")
_CATS_PL3 = _make("CATS_PL3")
_CATS_PL4 = _make("CATS_PL4")
_CATS_PL5 = _make("CATS_PL5")
_CATS_PL6 = _make("CATS_PL6")
_CATS_PL7 = _make("CATS_PL7")
_CATS_PL8 = _make("CATS_PL8")
_CATS_PL9 = _make("CATS_PL9")
_CATS_NN0 = _make("CATS_NN0")
_CATS_NN1 = _make("CATS_NN1")
_CATS_NN2 = _make("CATS_NN2")
_CATS_NN3 = _make("CATS_NN3")
_CATS_NN4 = _make("CATS_NN4")
_CATS_NN5 = _make("CATS_NN5")
_CATS_NN6 = _make("CATS_NN6")
_CATS_NN7 = _make("CATS_NN7")
_CATS_NN8 = _make("CATS_NN8")
_CATS_NN9 = _make("CATS_NN9")
_CATS_NL0 = _make("CATS_NL0")
_CATS_NL1 = _make("CATS_NL1")
_CATS_NL2 = _make("CATS_NL2")
_CATS_NL3 = _make("CATS_NL3")
_CATS_NL4 = _make("CATS_NL4")
_CATS_NL5 = _make("CATS_NL5")
_CATS_NL6 = _make("CATS_NL6")
_CATS_NL7 = _make("CATS_NL7")
_CATS_NL8 = _make("CATS_NL8")
_CATS_NL9 = _make("CATS_NL9")
_CATS_LL0 = _make("CATS_LL0")
_CATS_LL1 = _make("CATS_LL1")
_CATS_LL2 = _make("CATS_LL2")
_CATS_LL3 = _make("CATS_LL3")
_CATS_LL4 = _make("CATS_LL4")
_CATS_LL5 = _make("CATS_LL5")
_CATS_LL6 = _make("CATS_LL6")
_CATS_LL7 = _make("CATS_LL7")
_CATS_LL8 = _make("CATS_LL8")
_CATS_LL9 = _make("CATS_LL9")


# ---- public wrappers (name == function_name; declared -> 0 bonus) ----


def CATS_DD0(mol):
    """CATS_DD0 (CATS2D DD pharmacophore pair at topological distance 0)."""
    return _CATS_DD0(mol)


def CATS_DD1(mol):
    """CATS_DD1 (CATS2D DD pharmacophore pair at topological distance 1)."""
    return _CATS_DD1(mol)


def CATS_DD2(mol):
    """CATS_DD2 (CATS2D DD pharmacophore pair at topological distance 2)."""
    return _CATS_DD2(mol)


def CATS_DD3(mol):
    """CATS_DD3 (CATS2D DD pharmacophore pair at topological distance 3)."""
    return _CATS_DD3(mol)


def CATS_DD4(mol):
    """CATS_DD4 (CATS2D DD pharmacophore pair at topological distance 4)."""
    return _CATS_DD4(mol)


def CATS_DD5(mol):
    """CATS_DD5 (CATS2D DD pharmacophore pair at topological distance 5)."""
    return _CATS_DD5(mol)


def CATS_DD6(mol):
    """CATS_DD6 (CATS2D DD pharmacophore pair at topological distance 6)."""
    return _CATS_DD6(mol)


def CATS_DD7(mol):
    """CATS_DD7 (CATS2D DD pharmacophore pair at topological distance 7)."""
    return _CATS_DD7(mol)


def CATS_DD8(mol):
    """CATS_DD8 (CATS2D DD pharmacophore pair at topological distance 8)."""
    return _CATS_DD8(mol)


def CATS_DD9(mol):
    """CATS_DD9 (CATS2D DD pharmacophore pair at topological distance 9)."""
    return _CATS_DD9(mol)


def CATS_DA0(mol):
    """CATS_DA0 (CATS2D DA pharmacophore pair at topological distance 0)."""
    return _CATS_DA0(mol)


def CATS_DA1(mol):
    """CATS_DA1 (CATS2D DA pharmacophore pair at topological distance 1)."""
    return _CATS_DA1(mol)


def CATS_DA2(mol):
    """CATS_DA2 (CATS2D DA pharmacophore pair at topological distance 2)."""
    return _CATS_DA2(mol)


def CATS_DA3(mol):
    """CATS_DA3 (CATS2D DA pharmacophore pair at topological distance 3)."""
    return _CATS_DA3(mol)


def CATS_DA4(mol):
    """CATS_DA4 (CATS2D DA pharmacophore pair at topological distance 4)."""
    return _CATS_DA4(mol)


def CATS_DA5(mol):
    """CATS_DA5 (CATS2D DA pharmacophore pair at topological distance 5)."""
    return _CATS_DA5(mol)


def CATS_DA6(mol):
    """CATS_DA6 (CATS2D DA pharmacophore pair at topological distance 6)."""
    return _CATS_DA6(mol)


def CATS_DA7(mol):
    """CATS_DA7 (CATS2D DA pharmacophore pair at topological distance 7)."""
    return _CATS_DA7(mol)


def CATS_DA8(mol):
    """CATS_DA8 (CATS2D DA pharmacophore pair at topological distance 8)."""
    return _CATS_DA8(mol)


def CATS_DA9(mol):
    """CATS_DA9 (CATS2D DA pharmacophore pair at topological distance 9)."""
    return _CATS_DA9(mol)


def CATS_DP0(mol):
    """CATS_DP0 (CATS2D DP pharmacophore pair at topological distance 0)."""
    return _CATS_DP0(mol)


def CATS_DP1(mol):
    """CATS_DP1 (CATS2D DP pharmacophore pair at topological distance 1)."""
    return _CATS_DP1(mol)


def CATS_DP2(mol):
    """CATS_DP2 (CATS2D DP pharmacophore pair at topological distance 2)."""
    return _CATS_DP2(mol)


def CATS_DP3(mol):
    """CATS_DP3 (CATS2D DP pharmacophore pair at topological distance 3)."""
    return _CATS_DP3(mol)


def CATS_DP4(mol):
    """CATS_DP4 (CATS2D DP pharmacophore pair at topological distance 4)."""
    return _CATS_DP4(mol)


def CATS_DP5(mol):
    """CATS_DP5 (CATS2D DP pharmacophore pair at topological distance 5)."""
    return _CATS_DP5(mol)


def CATS_DP6(mol):
    """CATS_DP6 (CATS2D DP pharmacophore pair at topological distance 6)."""
    return _CATS_DP6(mol)


def CATS_DP7(mol):
    """CATS_DP7 (CATS2D DP pharmacophore pair at topological distance 7)."""
    return _CATS_DP7(mol)


def CATS_DP8(mol):
    """CATS_DP8 (CATS2D DP pharmacophore pair at topological distance 8)."""
    return _CATS_DP8(mol)


def CATS_DP9(mol):
    """CATS_DP9 (CATS2D DP pharmacophore pair at topological distance 9)."""
    return _CATS_DP9(mol)


def CATS_DN0(mol):
    """CATS_DN0 (CATS2D DN pharmacophore pair at topological distance 0)."""
    return _CATS_DN0(mol)


def CATS_DN1(mol):
    """CATS_DN1 (CATS2D DN pharmacophore pair at topological distance 1)."""
    return _CATS_DN1(mol)


def CATS_DN2(mol):
    """CATS_DN2 (CATS2D DN pharmacophore pair at topological distance 2)."""
    return _CATS_DN2(mol)


def CATS_DN3(mol):
    """CATS_DN3 (CATS2D DN pharmacophore pair at topological distance 3)."""
    return _CATS_DN3(mol)


def CATS_DN4(mol):
    """CATS_DN4 (CATS2D DN pharmacophore pair at topological distance 4)."""
    return _CATS_DN4(mol)


def CATS_DN5(mol):
    """CATS_DN5 (CATS2D DN pharmacophore pair at topological distance 5)."""
    return _CATS_DN5(mol)


def CATS_DN6(mol):
    """CATS_DN6 (CATS2D DN pharmacophore pair at topological distance 6)."""
    return _CATS_DN6(mol)


def CATS_DN7(mol):
    """CATS_DN7 (CATS2D DN pharmacophore pair at topological distance 7)."""
    return _CATS_DN7(mol)


def CATS_DN8(mol):
    """CATS_DN8 (CATS2D DN pharmacophore pair at topological distance 8)."""
    return _CATS_DN8(mol)


def CATS_DN9(mol):
    """CATS_DN9 (CATS2D DN pharmacophore pair at topological distance 9)."""
    return _CATS_DN9(mol)


def CATS_DL0(mol):
    """CATS_DL0 (CATS2D DL pharmacophore pair at topological distance 0)."""
    return _CATS_DL0(mol)


def CATS_DL1(mol):
    """CATS_DL1 (CATS2D DL pharmacophore pair at topological distance 1)."""
    return _CATS_DL1(mol)


def CATS_DL2(mol):
    """CATS_DL2 (CATS2D DL pharmacophore pair at topological distance 2)."""
    return _CATS_DL2(mol)


def CATS_DL3(mol):
    """CATS_DL3 (CATS2D DL pharmacophore pair at topological distance 3)."""
    return _CATS_DL3(mol)


def CATS_DL4(mol):
    """CATS_DL4 (CATS2D DL pharmacophore pair at topological distance 4)."""
    return _CATS_DL4(mol)


def CATS_DL5(mol):
    """CATS_DL5 (CATS2D DL pharmacophore pair at topological distance 5)."""
    return _CATS_DL5(mol)


def CATS_DL6(mol):
    """CATS_DL6 (CATS2D DL pharmacophore pair at topological distance 6)."""
    return _CATS_DL6(mol)


def CATS_DL7(mol):
    """CATS_DL7 (CATS2D DL pharmacophore pair at topological distance 7)."""
    return _CATS_DL7(mol)


def CATS_DL8(mol):
    """CATS_DL8 (CATS2D DL pharmacophore pair at topological distance 8)."""
    return _CATS_DL8(mol)


def CATS_DL9(mol):
    """CATS_DL9 (CATS2D DL pharmacophore pair at topological distance 9)."""
    return _CATS_DL9(mol)


def CATS_AA0(mol):
    """CATS_AA0 (CATS2D AA pharmacophore pair at topological distance 0)."""
    return _CATS_AA0(mol)


def CATS_AA1(mol):
    """CATS_AA1 (CATS2D AA pharmacophore pair at topological distance 1)."""
    return _CATS_AA1(mol)


def CATS_AA2(mol):
    """CATS_AA2 (CATS2D AA pharmacophore pair at topological distance 2)."""
    return _CATS_AA2(mol)


def CATS_AA3(mol):
    """CATS_AA3 (CATS2D AA pharmacophore pair at topological distance 3)."""
    return _CATS_AA3(mol)


def CATS_AA4(mol):
    """CATS_AA4 (CATS2D AA pharmacophore pair at topological distance 4)."""
    return _CATS_AA4(mol)


def CATS_AA5(mol):
    """CATS_AA5 (CATS2D AA pharmacophore pair at topological distance 5)."""
    return _CATS_AA5(mol)


def CATS_AA6(mol):
    """CATS_AA6 (CATS2D AA pharmacophore pair at topological distance 6)."""
    return _CATS_AA6(mol)


def CATS_AA7(mol):
    """CATS_AA7 (CATS2D AA pharmacophore pair at topological distance 7)."""
    return _CATS_AA7(mol)


def CATS_AA8(mol):
    """CATS_AA8 (CATS2D AA pharmacophore pair at topological distance 8)."""
    return _CATS_AA8(mol)


def CATS_AA9(mol):
    """CATS_AA9 (CATS2D AA pharmacophore pair at topological distance 9)."""
    return _CATS_AA9(mol)


def CATS_AP0(mol):
    """CATS_AP0 (CATS2D AP pharmacophore pair at topological distance 0)."""
    return _CATS_AP0(mol)


def CATS_AP1(mol):
    """CATS_AP1 (CATS2D AP pharmacophore pair at topological distance 1)."""
    return _CATS_AP1(mol)


def CATS_AP2(mol):
    """CATS_AP2 (CATS2D AP pharmacophore pair at topological distance 2)."""
    return _CATS_AP2(mol)


def CATS_AP3(mol):
    """CATS_AP3 (CATS2D AP pharmacophore pair at topological distance 3)."""
    return _CATS_AP3(mol)


def CATS_AP4(mol):
    """CATS_AP4 (CATS2D AP pharmacophore pair at topological distance 4)."""
    return _CATS_AP4(mol)


def CATS_AP5(mol):
    """CATS_AP5 (CATS2D AP pharmacophore pair at topological distance 5)."""
    return _CATS_AP5(mol)


def CATS_AP6(mol):
    """CATS_AP6 (CATS2D AP pharmacophore pair at topological distance 6)."""
    return _CATS_AP6(mol)


def CATS_AP7(mol):
    """CATS_AP7 (CATS2D AP pharmacophore pair at topological distance 7)."""
    return _CATS_AP7(mol)


def CATS_AP8(mol):
    """CATS_AP8 (CATS2D AP pharmacophore pair at topological distance 8)."""
    return _CATS_AP8(mol)


def CATS_AP9(mol):
    """CATS_AP9 (CATS2D AP pharmacophore pair at topological distance 9)."""
    return _CATS_AP9(mol)


def CATS_AN0(mol):
    """CATS_AN0 (CATS2D AN pharmacophore pair at topological distance 0)."""
    return _CATS_AN0(mol)


def CATS_AN1(mol):
    """CATS_AN1 (CATS2D AN pharmacophore pair at topological distance 1)."""
    return _CATS_AN1(mol)


def CATS_AN2(mol):
    """CATS_AN2 (CATS2D AN pharmacophore pair at topological distance 2)."""
    return _CATS_AN2(mol)


def CATS_AN3(mol):
    """CATS_AN3 (CATS2D AN pharmacophore pair at topological distance 3)."""
    return _CATS_AN3(mol)


def CATS_AN4(mol):
    """CATS_AN4 (CATS2D AN pharmacophore pair at topological distance 4)."""
    return _CATS_AN4(mol)


def CATS_AN5(mol):
    """CATS_AN5 (CATS2D AN pharmacophore pair at topological distance 5)."""
    return _CATS_AN5(mol)


def CATS_AN6(mol):
    """CATS_AN6 (CATS2D AN pharmacophore pair at topological distance 6)."""
    return _CATS_AN6(mol)


def CATS_AN7(mol):
    """CATS_AN7 (CATS2D AN pharmacophore pair at topological distance 7)."""
    return _CATS_AN7(mol)


def CATS_AN8(mol):
    """CATS_AN8 (CATS2D AN pharmacophore pair at topological distance 8)."""
    return _CATS_AN8(mol)


def CATS_AN9(mol):
    """CATS_AN9 (CATS2D AN pharmacophore pair at topological distance 9)."""
    return _CATS_AN9(mol)


def CATS_AL0(mol):
    """CATS_AL0 (CATS2D AL pharmacophore pair at topological distance 0)."""
    return _CATS_AL0(mol)


def CATS_AL1(mol):
    """CATS_AL1 (CATS2D AL pharmacophore pair at topological distance 1)."""
    return _CATS_AL1(mol)


def CATS_AL2(mol):
    """CATS_AL2 (CATS2D AL pharmacophore pair at topological distance 2)."""
    return _CATS_AL2(mol)


def CATS_AL3(mol):
    """CATS_AL3 (CATS2D AL pharmacophore pair at topological distance 3)."""
    return _CATS_AL3(mol)


def CATS_AL4(mol):
    """CATS_AL4 (CATS2D AL pharmacophore pair at topological distance 4)."""
    return _CATS_AL4(mol)


def CATS_AL5(mol):
    """CATS_AL5 (CATS2D AL pharmacophore pair at topological distance 5)."""
    return _CATS_AL5(mol)


def CATS_AL6(mol):
    """CATS_AL6 (CATS2D AL pharmacophore pair at topological distance 6)."""
    return _CATS_AL6(mol)


def CATS_AL7(mol):
    """CATS_AL7 (CATS2D AL pharmacophore pair at topological distance 7)."""
    return _CATS_AL7(mol)


def CATS_AL8(mol):
    """CATS_AL8 (CATS2D AL pharmacophore pair at topological distance 8)."""
    return _CATS_AL8(mol)


def CATS_AL9(mol):
    """CATS_AL9 (CATS2D AL pharmacophore pair at topological distance 9)."""
    return _CATS_AL9(mol)


def CATS_PP0(mol):
    """CATS_PP0 (CATS2D PP pharmacophore pair at topological distance 0)."""
    return _CATS_PP0(mol)


def CATS_PP1(mol):
    """CATS_PP1 (CATS2D PP pharmacophore pair at topological distance 1)."""
    return _CATS_PP1(mol)


def CATS_PP2(mol):
    """CATS_PP2 (CATS2D PP pharmacophore pair at topological distance 2)."""
    return _CATS_PP2(mol)


def CATS_PP3(mol):
    """CATS_PP3 (CATS2D PP pharmacophore pair at topological distance 3)."""
    return _CATS_PP3(mol)


def CATS_PP4(mol):
    """CATS_PP4 (CATS2D PP pharmacophore pair at topological distance 4)."""
    return _CATS_PP4(mol)


def CATS_PP5(mol):
    """CATS_PP5 (CATS2D PP pharmacophore pair at topological distance 5)."""
    return _CATS_PP5(mol)


def CATS_PP6(mol):
    """CATS_PP6 (CATS2D PP pharmacophore pair at topological distance 6)."""
    return _CATS_PP6(mol)


def CATS_PP7(mol):
    """CATS_PP7 (CATS2D PP pharmacophore pair at topological distance 7)."""
    return _CATS_PP7(mol)


def CATS_PP8(mol):
    """CATS_PP8 (CATS2D PP pharmacophore pair at topological distance 8)."""
    return _CATS_PP8(mol)


def CATS_PP9(mol):
    """CATS_PP9 (CATS2D PP pharmacophore pair at topological distance 9)."""
    return _CATS_PP9(mol)


def CATS_PN0(mol):
    """CATS_PN0 (CATS2D PN pharmacophore pair at topological distance 0)."""
    return _CATS_PN0(mol)


def CATS_PN1(mol):
    """CATS_PN1 (CATS2D PN pharmacophore pair at topological distance 1)."""
    return _CATS_PN1(mol)


def CATS_PN2(mol):
    """CATS_PN2 (CATS2D PN pharmacophore pair at topological distance 2)."""
    return _CATS_PN2(mol)


def CATS_PN3(mol):
    """CATS_PN3 (CATS2D PN pharmacophore pair at topological distance 3)."""
    return _CATS_PN3(mol)


def CATS_PN4(mol):
    """CATS_PN4 (CATS2D PN pharmacophore pair at topological distance 4)."""
    return _CATS_PN4(mol)


def CATS_PN5(mol):
    """CATS_PN5 (CATS2D PN pharmacophore pair at topological distance 5)."""
    return _CATS_PN5(mol)


def CATS_PN6(mol):
    """CATS_PN6 (CATS2D PN pharmacophore pair at topological distance 6)."""
    return _CATS_PN6(mol)


def CATS_PN7(mol):
    """CATS_PN7 (CATS2D PN pharmacophore pair at topological distance 7)."""
    return _CATS_PN7(mol)


def CATS_PN8(mol):
    """CATS_PN8 (CATS2D PN pharmacophore pair at topological distance 8)."""
    return _CATS_PN8(mol)


def CATS_PN9(mol):
    """CATS_PN9 (CATS2D PN pharmacophore pair at topological distance 9)."""
    return _CATS_PN9(mol)


def CATS_PL0(mol):
    """CATS_PL0 (CATS2D PL pharmacophore pair at topological distance 0)."""
    return _CATS_PL0(mol)


def CATS_PL1(mol):
    """CATS_PL1 (CATS2D PL pharmacophore pair at topological distance 1)."""
    return _CATS_PL1(mol)


def CATS_PL2(mol):
    """CATS_PL2 (CATS2D PL pharmacophore pair at topological distance 2)."""
    return _CATS_PL2(mol)


def CATS_PL3(mol):
    """CATS_PL3 (CATS2D PL pharmacophore pair at topological distance 3)."""
    return _CATS_PL3(mol)


def CATS_PL4(mol):
    """CATS_PL4 (CATS2D PL pharmacophore pair at topological distance 4)."""
    return _CATS_PL4(mol)


def CATS_PL5(mol):
    """CATS_PL5 (CATS2D PL pharmacophore pair at topological distance 5)."""
    return _CATS_PL5(mol)


def CATS_PL6(mol):
    """CATS_PL6 (CATS2D PL pharmacophore pair at topological distance 6)."""
    return _CATS_PL6(mol)


def CATS_PL7(mol):
    """CATS_PL7 (CATS2D PL pharmacophore pair at topological distance 7)."""
    return _CATS_PL7(mol)


def CATS_PL8(mol):
    """CATS_PL8 (CATS2D PL pharmacophore pair at topological distance 8)."""
    return _CATS_PL8(mol)


def CATS_PL9(mol):
    """CATS_PL9 (CATS2D PL pharmacophore pair at topological distance 9)."""
    return _CATS_PL9(mol)


def CATS_NN0(mol):
    """CATS_NN0 (CATS2D NN pharmacophore pair at topological distance 0)."""
    return _CATS_NN0(mol)


def CATS_NN1(mol):
    """CATS_NN1 (CATS2D NN pharmacophore pair at topological distance 1)."""
    return _CATS_NN1(mol)


def CATS_NN2(mol):
    """CATS_NN2 (CATS2D NN pharmacophore pair at topological distance 2)."""
    return _CATS_NN2(mol)


def CATS_NN3(mol):
    """CATS_NN3 (CATS2D NN pharmacophore pair at topological distance 3)."""
    return _CATS_NN3(mol)


def CATS_NN4(mol):
    """CATS_NN4 (CATS2D NN pharmacophore pair at topological distance 4)."""
    return _CATS_NN4(mol)


def CATS_NN5(mol):
    """CATS_NN5 (CATS2D NN pharmacophore pair at topological distance 5)."""
    return _CATS_NN5(mol)


def CATS_NN6(mol):
    """CATS_NN6 (CATS2D NN pharmacophore pair at topological distance 6)."""
    return _CATS_NN6(mol)


def CATS_NN7(mol):
    """CATS_NN7 (CATS2D NN pharmacophore pair at topological distance 7)."""
    return _CATS_NN7(mol)


def CATS_NN8(mol):
    """CATS_NN8 (CATS2D NN pharmacophore pair at topological distance 8)."""
    return _CATS_NN8(mol)


def CATS_NN9(mol):
    """CATS_NN9 (CATS2D NN pharmacophore pair at topological distance 9)."""
    return _CATS_NN9(mol)


def CATS_NL0(mol):
    """CATS_NL0 (CATS2D NL pharmacophore pair at topological distance 0)."""
    return _CATS_NL0(mol)


def CATS_NL1(mol):
    """CATS_NL1 (CATS2D NL pharmacophore pair at topological distance 1)."""
    return _CATS_NL1(mol)


def CATS_NL2(mol):
    """CATS_NL2 (CATS2D NL pharmacophore pair at topological distance 2)."""
    return _CATS_NL2(mol)


def CATS_NL3(mol):
    """CATS_NL3 (CATS2D NL pharmacophore pair at topological distance 3)."""
    return _CATS_NL3(mol)


def CATS_NL4(mol):
    """CATS_NL4 (CATS2D NL pharmacophore pair at topological distance 4)."""
    return _CATS_NL4(mol)


def CATS_NL5(mol):
    """CATS_NL5 (CATS2D NL pharmacophore pair at topological distance 5)."""
    return _CATS_NL5(mol)


def CATS_NL6(mol):
    """CATS_NL6 (CATS2D NL pharmacophore pair at topological distance 6)."""
    return _CATS_NL6(mol)


def CATS_NL7(mol):
    """CATS_NL7 (CATS2D NL pharmacophore pair at topological distance 7)."""
    return _CATS_NL7(mol)


def CATS_NL8(mol):
    """CATS_NL8 (CATS2D NL pharmacophore pair at topological distance 8)."""
    return _CATS_NL8(mol)


def CATS_NL9(mol):
    """CATS_NL9 (CATS2D NL pharmacophore pair at topological distance 9)."""
    return _CATS_NL9(mol)


def CATS_LL0(mol):
    """CATS_LL0 (CATS2D LL pharmacophore pair at topological distance 0)."""
    return _CATS_LL0(mol)


def CATS_LL1(mol):
    """CATS_LL1 (CATS2D LL pharmacophore pair at topological distance 1)."""
    return _CATS_LL1(mol)


def CATS_LL2(mol):
    """CATS_LL2 (CATS2D LL pharmacophore pair at topological distance 2)."""
    return _CATS_LL2(mol)


def CATS_LL3(mol):
    """CATS_LL3 (CATS2D LL pharmacophore pair at topological distance 3)."""
    return _CATS_LL3(mol)


def CATS_LL4(mol):
    """CATS_LL4 (CATS2D LL pharmacophore pair at topological distance 4)."""
    return _CATS_LL4(mol)


def CATS_LL5(mol):
    """CATS_LL5 (CATS2D LL pharmacophore pair at topological distance 5)."""
    return _CATS_LL5(mol)


def CATS_LL6(mol):
    """CATS_LL6 (CATS2D LL pharmacophore pair at topological distance 6)."""
    return _CATS_LL6(mol)


def CATS_LL7(mol):
    """CATS_LL7 (CATS2D LL pharmacophore pair at topological distance 7)."""
    return _CATS_LL7(mol)


def CATS_LL8(mol):
    """CATS_LL8 (CATS2D LL pharmacophore pair at topological distance 8)."""
    return _CATS_LL8(mol)


def CATS_LL9(mol):
    """CATS_LL9 (CATS2D LL pharmacophore pair at topological distance 9)."""
    return _CATS_LL9(mol)


ALL_DESCRIPTOR_NAMES = (
    "CATS_DD0",
    "CATS_DD1",
    "CATS_DD2",
    "CATS_DD3",
    "CATS_DD4",
    "CATS_DD5",
    "CATS_DD6",
    "CATS_DD7",
    "CATS_DD8",
    "CATS_DD9",
    "CATS_DA0",
    "CATS_DA1",
    "CATS_DA2",
    "CATS_DA3",
    "CATS_DA4",
    "CATS_DA5",
    "CATS_DA6",
    "CATS_DA7",
    "CATS_DA8",
    "CATS_DA9",
    "CATS_DP0",
    "CATS_DP1",
    "CATS_DP2",
    "CATS_DP3",
    "CATS_DP4",
    "CATS_DP5",
    "CATS_DP6",
    "CATS_DP7",
    "CATS_DP8",
    "CATS_DP9",
    "CATS_DN0",
    "CATS_DN1",
    "CATS_DN2",
    "CATS_DN3",
    "CATS_DN4",
    "CATS_DN5",
    "CATS_DN6",
    "CATS_DN7",
    "CATS_DN8",
    "CATS_DN9",
    "CATS_DL0",
    "CATS_DL1",
    "CATS_DL2",
    "CATS_DL3",
    "CATS_DL4",
    "CATS_DL5",
    "CATS_DL6",
    "CATS_DL7",
    "CATS_DL8",
    "CATS_DL9",
    "CATS_AA0",
    "CATS_AA1",
    "CATS_AA2",
    "CATS_AA3",
    "CATS_AA4",
    "CATS_AA5",
    "CATS_AA6",
    "CATS_AA7",
    "CATS_AA8",
    "CATS_AA9",
    "CATS_AP0",
    "CATS_AP1",
    "CATS_AP2",
    "CATS_AP3",
    "CATS_AP4",
    "CATS_AP5",
    "CATS_AP6",
    "CATS_AP7",
    "CATS_AP8",
    "CATS_AP9",
    "CATS_AN0",
    "CATS_AN1",
    "CATS_AN2",
    "CATS_AN3",
    "CATS_AN4",
    "CATS_AN5",
    "CATS_AN6",
    "CATS_AN7",
    "CATS_AN8",
    "CATS_AN9",
    "CATS_AL0",
    "CATS_AL1",
    "CATS_AL2",
    "CATS_AL3",
    "CATS_AL4",
    "CATS_AL5",
    "CATS_AL6",
    "CATS_AL7",
    "CATS_AL8",
    "CATS_AL9",
    "CATS_PP0",
    "CATS_PP1",
    "CATS_PP2",
    "CATS_PP3",
    "CATS_PP4",
    "CATS_PP5",
    "CATS_PP6",
    "CATS_PP7",
    "CATS_PP8",
    "CATS_PP9",
    "CATS_PN0",
    "CATS_PN1",
    "CATS_PN2",
    "CATS_PN3",
    "CATS_PN4",
    "CATS_PN5",
    "CATS_PN6",
    "CATS_PN7",
    "CATS_PN8",
    "CATS_PN9",
    "CATS_PL0",
    "CATS_PL1",
    "CATS_PL2",
    "CATS_PL3",
    "CATS_PL4",
    "CATS_PL5",
    "CATS_PL6",
    "CATS_PL7",
    "CATS_PL8",
    "CATS_PL9",
    "CATS_NN0",
    "CATS_NN1",
    "CATS_NN2",
    "CATS_NN3",
    "CATS_NN4",
    "CATS_NN5",
    "CATS_NN6",
    "CATS_NN7",
    "CATS_NN8",
    "CATS_NN9",
    "CATS_NL0",
    "CATS_NL1",
    "CATS_NL2",
    "CATS_NL3",
    "CATS_NL4",
    "CATS_NL5",
    "CATS_NL6",
    "CATS_NL7",
    "CATS_NL8",
    "CATS_NL9",
    "CATS_LL0",
    "CATS_LL1",
    "CATS_LL2",
    "CATS_LL3",
    "CATS_LL4",
    "CATS_LL5",
    "CATS_LL6",
    "CATS_LL7",
    "CATS_LL8",
    "CATS_LL9",
)
