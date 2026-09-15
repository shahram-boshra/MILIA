# `matrix_spectral` — eigenvalue / matrix-spectral descriptors

First-party descriptor plugin shipped with MILIA (descriptor programme, **Pace 5 / v1.8.0**).
177 `PLUGIN-NATIVE` eigenvalue/matrix-spectral descriptors, implemented from primary literature and
validated against `mordredcommunity` (rel-err ≤ 1e-4; in practice ≤ 6e-8).

## Contents (177 descriptors)

| Block | Descriptors | Count | Reference |
|---|---|---:|---|
| `AdjacencyMatrix` | `SpAbs/SpMax/SpDiam/SpAD/SpMAD/LogEE/VE1-3/VR1-3` `_A` | 12 | Todeschini & Consonni |
| `DistanceMatrix` | same spectra `_D` | 12 | Todeschini & Consonni |
| `DetourMatrix` | + `SM1` `_Dt` | 13 | Trinajstić |
| `BaryszMatrix` | spectra × 8 weightings (`Z/m/v/se/pe/are/p/i`) `_Dz*` | 104 | Barysz et al. 1983 |
| `BCUT` | Burden eigenvalues × 12 weightings × hi/lo | 24 | Pearlman & Smith 1999 |
| `MolecularId` | `MID/AMID` × {any, hetero, C, N, O, X} | 12 | Randić 1984 |

Total: **177** (2D; `requires_3d: false`; no optional extra).

## Design

* **Shared eigen/spectral engine.** `SpAbs` (graph energy), `SpMax` (leading eigenvalue), `SpDiam`,
  `SpAD`/`SpMAD`, `LogEE` (Estrada-like log-sum-exp), `SM1` (trace), `VE1-3` (leading-eigenvector
  coefficient sums), `VR1-3` (Randić eigenvector-based) — computed from `numpy` eigendecomposition.
* **Matrix constructions (native).** Adjacency/distance from RDKit; detour (longest-simple-path) via
  pure-stdlib DFS; Barysz = Floyd-Warshall on property-weighted edges `w = C²/(P_i·P_j·π)`; Burden
  matrix (`0.001` non-bonded, `bond/10 +0.01` terminal, property diagonal) for BCUT; Randić
  weighted-path DFS for MolecularId. **No networkx.**
* **Requires connected.** Every descriptor returns `NaN` on a disconnected graph (and on any invalid
  input / unparameterized element) — never raises.
* **Purity & naming.** RDKit + stdlib + numpy only; deterministic. `name == function_name`; the 24
  BCUT names are hyphen-sanitized to valid identifiers (`BCUTc-1h` → `BCUTc_1h`, canonical in the
  description), so all 177 register exactly once (0 bonus discoveries).

## Scoping notes

* **De-dup:** mordred BCUT is kept — it uses a different Burden convention than the HAVE RDKit
  `BCUT2D` baseline (distinct values, verified), so it is not a duplicate. `DetourIndex` (the scalar)
  is **not** here — it shipped in `walk_path_information` (Pace 4); this plugin carries only the 13
  detour-matrix *spectra*.
