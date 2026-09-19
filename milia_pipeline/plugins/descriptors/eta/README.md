# `eta` — Extended Topochemical Atom (ETA) descriptors

First-party descriptor plugin shipped with MILIA (descriptor programme, **Pace 8 / v1.11.0**).
45 `PLUGIN-NATIVE` second-generation ETA indices (Roy & Ghosh) computed RDKit-natively and
validated **bit-exact** against `mordredcommunity` (rel-err 0; 855/855 on the panel).

## Contents (45 descriptors)

| Group | Descriptors |
|---|---|
| Core count (α) | `ETA_alpha`, `AETA_alpha` |
| Shape | `ETA_shape_p/y/x` |
| VEM count (β) | `ETA_beta`, `ETA_beta_s`, `ETA_beta_ns`, `ETA_beta_ns_d` (+ `AETA_*`) |
| Composite (η) | `ETA_eta`, `_L`, `_R`, `_RL` (+ `AETA_*`) |
| Functionality (η_F) | `ETA_eta_F`, `_FL` (+ `AETA_*`) |
| Branching (η_B) | `ETA_eta_B`, `_BR` (+ `AETA_*`) |
| Δα | `ETA_dAlpha_A/B` |
| Epsilon (ε) | `ETA_epsilon_1..5` |
| Δε | `ETA_dEpsilon_A/B/C/D` |
| Δβ | `ETA_dBeta`, `AETA_dBeta` |
| ψ | `ETA_psi_1`, `ETA_dPsi_A/B` |

Total **45** (2D; `requires_3d: false`; `category: topological`; `block: ExtendedTopochemicalAtom`).

## Design

* **Native from literature.** Per-atom ETA primitives — core count `α = (Z−Zv)/(Zv·(PN−1))`,
  electronegativity `ε = 0.3·Zv − α`, VEM `β` (sigma/nonsigma/delta), and `γ = α/β` — computed
  directly from RDKit atom/bond data (Roy & Ghosh formulas). The composite index sums
  `√(γ_i·γ_j / r_ij²)` over the topological-distance matrix; functionality/branching/Δα use an
  all-carbon single-bond **reference** graph, and ε-3/ε-4 use the reference and **saturated**
  skeletons — all built exactly as the oracle does (kekulized, hydrogen-suppressed base).
* **Block-cache.** One memoised `_eta_block(mol)` (`functools.lru_cache(maxsize=1)` keyed on the
  `Mol`) computes all 45 once; the 45 wrappers index the cache.
* **Oracle-exact.** Reproduces `mordred.ExtendedTopochemicalAtom` bit-for-bit on the panel,
  including aromatic heterocycles (kekulized reference/saturated graphs) and the internal
  `require_connected` behaviour.
* **Purity.** RDKit + NumPy + stdlib only; `name == function_name` (all valid identifiers); all
  45 declared → 0 bonus discoveries.

## Scoping notes

* **Re-scoped Pace 8.** The Blueprint's original Pace-8 families (edge-adjacency spectra + extended-
  rank Burden) are **Dragon-specific and absent from mordred** (its `BCUT` is only `1h/1l`, already
  shipped in `matrix_spectral`), so they carry no mordred oracle and were **deferred to Pace 11**
  (PaDEL/CDK-residual). ETA is the largest clean, mordred-backed, zero-collision 2D family still
  unshipped, and became Pace 8.
* **No collision.** `ETA_*`/`AETA_*` names are disjoint from HAVE and every prior pace.
* **Disconnected / empty.** Multi-fragment or 0-atom molecules return NaN for all 45 (oracle
  `require_connected`).
