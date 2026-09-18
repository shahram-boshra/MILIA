# `estate_atomtype` — Kier-Hall E-State atom-type descriptors

First-party descriptor plugin shipped with MILIA (descriptor programme, **Pace 7 / v1.10.0**).
316 `PLUGIN-NATIVE` electrotopological-state (E-State) atom-type descriptors computed
RDKit-natively and validated against `mordredcommunity` (rel-err ≤ 1e-4; actual 0 on every
valid panel molecule).

## Contents (316 descriptors)

79 Kier-Hall atom types × 4 aggregations:

| Prefix | Aggregation | Value | Count |
|---|---|---|---:|
| `N<type>` | count | number of atoms of the E-State atom type | 79 |
| `S<type>` | sum | Σ E-State value over atoms of that type | 79 |
| `MAX<type>` | max | max E-State value among atoms of that type (NaN if none) | 79 |
| `MIN<type>` | min | min E-State value among atoms of that type (NaN if none) | 79 |

Atom types (e.g. `sCH3`, `dO`, `aaCH`, `aasC`, `ssNH`, `sF`, …) follow the Kier-Hall
nomenclature. Total: **316** (2D; `requires_3d: false`; `category: electronic`;
`block: AtomTypeEState`).

## Design

* **RDKit-native.** A single memoised block function `_estate_block(mol)` computes, once per
  molecule, the (count, sum, max, min) of `rdkit.Chem.EState.EStateIndices` grouped by
  `rdkit.Chem.EState.AtomTypes.TypeAtoms` labels (the H-suppressed Kier-Hall graph). The 316
  public wrappers are thin indexers into that cache (`functools.lru_cache(maxsize=1)` keyed on
  the `Mol`, single-slot self-eviction: N indexers → 1 compute).
* **Oracle-exact.** RDKit's 79-label E-State vocabulary is identical to the `mordredcommunity`
  oracle (verified: empty symmetric difference), so all 316 reproduce mordred at rel-err 0 on
  every valid molecule of the panel.
* **Purity.** RDKit + stdlib only; `name == function_name` (all valid identifiers); all 316
  declared → 0 bonus discoveries.

## Scoping notes

* **Zero de-dup / no collision.** These atom-typed sums/counts are distinct in name and value
  from MILIA's HAVE aggregate E-State indices (`Max/Min{,Abs}EStateIndex`, RDKit `ELECTRONIC`
  baseline). No collision with HAVE or any prior pace.
* **MoeType excluded from Pace 7.** All 53 `mordred.MoeType` descriptors
  (`LabuteASA`, `PEOE_VSA*`, `SMR_VSA*`, `SlogP_VSA*`, `EState_VSA*`, `VSA_EState*`) are already
  in MILIA's HAVE registry (RDKit-canonical bins) → not shipped (name-uniqueness invariant).
* **Empty molecule.** A 0-atom (invalid) molecule returns NaN for **all** 316 aggregations
  (MILIA purity contract). The oracle is internally inconsistent there (NaN for counts, 0 for
  sums); the frozen-snapshot test compares valid molecules exactly and asserts all-NaN for the
  empty case.
