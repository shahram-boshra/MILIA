# `cats2d` — CATS2D topological pharmacophore-pair descriptors

First-party descriptor plugin shipped with MILIA (descriptor programme, **Pace 7 / v1.10.0**).
150 `PLUGIN-NATIVE` CATS2D descriptors (Chemically Advanced Template Search; Schneider et al.,
*Angew. Chem. Int. Ed.* 1999) validated **bit-exact** against the peer-reviewed **PyBioMed**
reference implementation (Dong et al., *J. Cheminformatics* 2018) at raw-count scaling
(2100/2100, 0 mismatches).

## Contents (150 descriptors)

15 canonical potential-pharmacophore-point (PPP) pairs × 10 topological-distance bins (0–9),
**raw counts**:

| | D | A | P | N | L |
|---|---|---|---|---|---|
| **pairs** | DD, DA, DP, DN, DL | AA, AP, AN, AL | PP, PN, PL | NN, NL | LL |

Descriptor names: `CATS_<pair><k>` for `k = 0..9` (e.g. `CATS_DA3`, `CATS_LL0`). PPP types:
`D` H-bond donor, `A` H-bond acceptor, `P` positive, `N` negative, `L` lipophilic.
Total **150** (2D; `requires_3d: false`; `category: fragments`; `block: CATS2D`).

## Design

* **Native from literature.** PPP typing via the published Schneider CATS SMARTS (donor `[OH]`,
  `[#7H,#7H2]`; acceptor `[O]`, `[#7H0]`; positive `[*+]`, `[#7H2]`; negative `[*-]` + COOH/
  SOOH/POOH acids; lipophilic `[Cl,Br,I]`, dialkyl-`S`, **plus** the graph-search L =
  carbon whose only non-H neighbours are carbon). On the heavy-atom graph (`RemoveHs`), for
  each atom pair at topological distance *k* every distinct canonical PPP-pair code (priority
  `D<A<P<N<L`) is counted once; self-pairs populate distance 0.
* **Block-cache.** A single memoised `_cats_block(mol)` (`functools.lru_cache(maxsize=1)` keyed
  on the `Mol`) builds all 150 counts once; the 150 wrappers index the cache.
* **Oracle-exact.** Reproduces PyBioMed `CATS2D(scale=1)` exactly on the panel. PyBioMed
  requires a legacy scipy shim, so it is **not** a MILIA runtime/test dependency — the frozen
  snapshot is the authoritative gate.
* **Purity.** RDKit + stdlib only; `name == function_name` (all valid identifiers); all 150
  declared → 0 bonus discoveries.

## Scoping notes

* **Raw counts only.** The base variant (`scale=1`, integer counts) is shipped; the sum- and
  pair-normalised scalings (PyBioMed `scale=2/3`) are intentionally deferred to avoid 3× scope
  inflation and keep a deterministic integer-valued base.
* **No collision.** `CATS_*` names are disjoint from HAVE and every prior pace (in particular
  the unrelated 3D `USRCAT_*` shape descriptors from Pace 1). Not in RDKit or mordred.
* **CATS not in the exhaustive list.** `Exhaustive_Molecular_Descriptors.md` gains a
  `B.CATS2D (150)` block in this pace.
* **Empty molecule.** A 0-atom (invalid) molecule returns NaN for all 150 (MILIA purity
  contract).
