# `walk_path_information` — walk/path + detour + information-content descriptors

First-party descriptor plugin shipped with MILIA (descriptor programme, **Pace 4 / v1.7.0**).
86 `PLUGIN-NATIVE` topological descriptors implemented from primary literature and validated
**bit-exact** against `mordredcommunity` (`rtol <= 1e-4`).

## Contents (86 descriptors)

| Block | Descriptors | Count | Primary reference |
|---|---|---:|---|
| `WalkCount` | `MWC01-10`, `TMWC10`, `SRW02-10`, `TSRW10` | 21 | Rücker & Rücker 1993; Harary 1969 |
| `PathCount` | `MPC2-10`, `TMPC10`, `piPC1-10`, `TpiPC10` | 21 | Randić 1979 |
| `DetourIndex` | `DetourIndex` | 1 | Trinajstić, *Chemical Graph Theory* |
| `InformationContent` | `IC/TIC/SIC/BIC/CIC/MIC/ZMIC` × orders 0-5 | 42 | Bonchev & Trinajstić 1977; Basak 1999 |
| `VertexAdjacencyInformation` | `VAdjMat` | 1 | Todeschini & Consonni, *Handbook* |

Total: **86** (2D; `requires_3d: false`; no optional extra).

## Design

* **Native from literature.** Walk counts from powers of the adjacency matrix; path counts by
  self-avoiding path enumeration with bond-order (π) weighting; information content by
  BFS-tree atom-equivalence-class Shannon entropy; detour index from the longest-simple-path
  matrix (pure-stdlib DFS — no networkx).
* **Oracle-gated.** All 86 reproduce `mordredcommunity` bit-exact on the validation panel.
* **Purity.** Imports only RDKit + stdlib + numpy. Deterministic (2D, no RNG). NaN on any
  invalid molecule / undefined value — never raises. `name == function_name` (single-registration
  invariant; all names are valid identifiers).

## Conventions (verified against the oracle)

* `WalkCount` / `PathCount` / `DetourIndex` / `VAdjMat` operate on the **heavy-atom** graph.
* `InformationContent` operates on the **H-included, kekulized** graph (the oracle's default),
  so its atom count and bond-order sums include hydrogens.
* `MWC01` = number of edges; walk/path counts of order ≥ 2 use the `log(x + 1)` transform.
* `DetourIndex` is longest-simple-path (NP-hard); a DFS **step budget** returns `NaN` for
  pathological fused-polycyclics rather than hanging — drug-like molecules stay far under it.

## Scoping notes

* `MWC03 = 2·Zagreb2` is a known affine **correlation** (distinct value, not an exact
  duplicate) — both are shipped, as DRAGON/mordred do.
* The 13 detour-matrix **eigenvalue-spectra** (`SpAbs_Dt`/`VE*_Dt`/`VR*_Dt`/`SM1_Dt`/`LogEE_Dt`)
  are deferred to the dedicated eigenvalue/matrix-spectra pace (Pace 5).
