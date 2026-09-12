# `topological_connectivity` — topological / connectivity indices

First-party descriptor plugin shipped with MILIA (descriptor programme, **Pace 3 / v1.6.0**).
107 `PLUGIN-NATIVE` topological/connectivity descriptors implemented from primary literature and
validated against an oracle: `mordredcommunity` for the 104 Mordred-defined descriptors
(reproduced **bit-exact**, `rtol <= 1e-4`) and published reference values for the 3 Schultz indices.

## Contents (107 descriptors)

| Block | Descriptors | Count | Primary reference |
|---|---|---:|---|
| `WienerIndex` | `WPath`, `WPol` | 2 | Wiener 1947 |
| `ZagrebIndex` | `Zagreb1/2`, `mZagreb1/2` | 4 | Gutman & Trinajstić 1972 |
| `ABCIndex` | `ABC`, `ABCGG` | 2 | Estrada 1998; Graovac-Ghorbani 2016 |
| `EccentricConnectivity` | `ECIndex` | 1 | Sharma, Goswami & Madan 1997 |
| `TopologicalIndex` | `Diameter`, `Radius`, `TopoShapeIndex`, `PetitjeanIndex` | 4 | Petitjean 1992 |
| `Schultz` | `MTI`, `SchultzIndex`, `ModSchultzIndex` | 3 | Schultz 1989 |
| `TopologicalCharge` | `GGI1-10`, `JGI1-10`, `JGT10` | 21 | Gálvez et al. 1994 |
| `MolecularDistanceEdge` | `MDEC/MDEO/MDEN` | 19 | Liu, Cai & Yao 1998 |
| `Chi` | Kier-Hall path/cluster/path-cluster/chain (simple + valence) | 51 | Kier & Hall 1976/1986 |

Total: **107** (2D; `requires_3d: false`; no optional extra).

## Design

* **Native from literature.** Every descriptor is computed from RDKit graph primitives
  (adjacency/distance matrices, subgraph enumeration, Kier-Hall δ/δv deltas) per its primary
  source — no Mordred/PaDEL calls at runtime.
* **Oracle-gated.** The 104 Mordred-defined descriptors reproduce `mordredcommunity` bit-exact;
  `MTI`/`SchultzIndex`/`ModSchultzIndex` are gated against published values (benzene 132/54/54;
  propane 5/3) and the identity `MTI = Zagreb1 + 2·SchultzIndex`.
* **Purity.** Imports only RDKit + stdlib + numpy. Deterministic (2D, no RNG). NaN on any invalid
  molecule / undefined value — never raises.
* **Naming.** Registered `name` **equals** `function_name` (a valid Python identifier, e.g. `Xp_0d`,
  `MDEC_11`) — the single-registration invariant. The canonical Mordred string (`Xp-0d`, `MDEC-11`) is
  preserved in each descriptor's `description`. All 107 register exactly once (**0 bonus discoveries**);
  using the hyphenated Mordred string as `name` would make the undeclared-scan double-register each.

## Scoping notes

* **De-dup (FractionCSP3-style):** the 5 valence-path low orders `Xp-0dv…Xp-4dv` are **not**
  shipped — they equal the existing HAVE RDKit `Chi0v…Chi4v` exactly (verified bit-identical on a
  heteroatom-rich panel). The simple-path `Xp-0d…Xp-4d` use a different normalisation than HAVE's
  `Chi_kn` and are kept.
* **Deferred to later paces:** the ~166 eigenvalue/matrix-spectra descriptors (Barysz / adjacency /
  distance / detour matrix spectra, BCUT) — a distinct methodological family — and any descriptor
  lacking a Mordred/RDKit oracle beyond the deliberately-included Schultz set.
