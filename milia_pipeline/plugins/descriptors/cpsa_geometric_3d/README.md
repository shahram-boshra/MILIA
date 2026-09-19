# `cpsa_geometric_3d` — CPSA, gravitational & geometrical 3D descriptors

First-party descriptor plugin shipped with MILIA (descriptor programme, **Pace 9 / v1.12.0**).
50 `PLUGIN-NATIVE` conformer-dependent 3D descriptors computed on the MILIA-core conformer
and validated **bit-exact** against `mordredcommunity` (rel-err 0; 600/600 on the panel,
verified under both rdkit 2025.3.6 and 2026.03.6).

## Contents (50 descriptors)

| Block | n | Descriptors |
|---|---:|---|
| **CPSA** | 42 | `PNSA1–5`, `PPSA1–5`, `DPSA1–5`, `FNSA1–5`, `FPSA1–5`, `WNSA1–5`, `WPSA1–5`, `RNCG`, `RPCG`, `RNCS`, `RPCS`, `TASA`, `RASA`, `RPSA` |
| **GravitationalIndex** | 4 | `GRAV`, `GRAVH`, `GRAVp`, `GRAVHp` |
| **GeometricalIndex** | 4 | `GeomDiameter`, `GeomRadius`, `GeomShapeIndex`, `GeomPetitjeanIndex` |

`category: geometric`, `requires_3d: true`, `dependencies: []`.

## Design

* **Conformer.** All 50 use the conformer embedded by the MILIA calculator —
  `AddHs → EmbedMolecule(randomSeed=42) → MMFFOptimizeMolecule` — read via `GetConformer(-1)`.
* **CPSA** (Stanton & Jurs, *Anal. Chem.* 1990, 62, 2323): atomic solvent-accessible surface
  area from a **Shrake–Rupley dot tessellation** over a level-5 icosphere (vdW + 1.4 Å solvent
  radius), combined with **RDKit Gasteiger** partial charges — negative/positive/fractional/
  surface-weighted surface areas, relative charges, and hydrophobic/polar surface areas. The
  tessellation mesh and SASA routine are transcribed verbatim from the oracle (deterministic).
* **GravitationalIndex**: mass-weighted Σ mᵢmⱼ / rᵢⱼ² over all pairs (`GRAV`) or bonded pairs
  (`GRAVp`), heavy-atom-only or including H (`GRAVH*`).
* **GeometricalIndex**: geometric diameter/radius (max/min eccentricity of the 3D distance
  matrix), shape index `(D−R)/R`, and Petitjean index `(D−R)/D`.
* **Block-cache.** One memoised `_block(mol)` computes all 50 once; the 50 wrappers index it.
* **Purity.** RDKit + NumPy + stdlib only; `name == function_name` (valid identifiers); all 50
  declared → 0 bonus. Missing conformer / None / degenerate → NaN, never raise.

## Scoping notes

* **Not dependency-bound.** RDKit provides SASA (native tessellation), Gasteiger charges, and
  masses, so this ships in base like `addcore_3d` (`dependencies: []`) — the Blueprint's original
  "DEP-BOUND `[descriptors-3d]`" assumption did not hold.
* **De-dup (50, not 55).** The mordred CPSA **`TPSA`** (3D total polar surface area) is **not
  shipped** — its name collides with the HAVE RDKit **topological** `TPSA` (registry
  name-uniqueness); it is still computed internally because `RPSA = TPSA / ΣSASA`.
  **`MomentOfInertia`** (`MOMI-X/Y/Z` = HAVE `PMI1/2/3`, verified identical) and **`PBF`**
  (= HAVE `PBF`) are excluded as duplicates.
* **Version-robust oracle.** The frozen snapshot stores the exact conformer (MolBlock +
  full-precision coordinates) plus the mordred values, so the test reconstructs the geometry
  rather than re-embedding — bit-exact regardless of the RDKit build.
