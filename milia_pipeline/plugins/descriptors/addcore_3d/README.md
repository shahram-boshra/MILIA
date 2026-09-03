# `addcore_3d` — ADD-CORE RDKit 3D descriptor blocks

First-party descriptor plugin shipped with MILIA (descriptor programme, **Pace 1 / v1.4.0**).
It exposes the RDKit-native descriptor blocks that are **not** part of the MILIA `HAVE`
baseline as individually-named scalar descriptors, under the standard zero-core-modification
plugin contract.

## Contents (1,019 descriptors)

| Block | Count | Category | RDKit source | 3D? | Primary reference |
|---|---:|---|---|:--:|---|
| `WHIM_1..114` | 114 | geometric | `CalcWHIM` | yes | Todeschini & Gramatica 1997 |
| `GETAWAY_1..273` | 273 | geometric | `CalcGETAWAY` | yes | Consonni, Todeschini & Pavan 2002 |
| `RDF_1..210` | 210 | geometric | `CalcRDF` | yes | Hemmer, Steinhauer & Gasteiger 1999 |
| `MoRSE_1..224` | 224 | geometric | `CalcMORSE` | yes | Schuur, Selzer & Gasteiger 1996 |
| `AUTOCORR3D_1..80` | 80 | geometric | `CalcAUTOCORR3D` | yes | Moreau & Broto 1980 (3D) |
| `USR_1..12` | 12 | geometric | `GetUSR` | yes | Ballester & Richards 2007 |
| `USRCAT_1..60` | 60 | geometric | `GetUSRCAT` | yes | Schreyer & Blundell 2012 |
| `MQN_1..42` | 42 | constitutional | `MQNs_` | no | Nguyen, Blum et al. 2009 |
| `OxidationNumber_{min,max,mean,sum}` | 4 | constitutional | `CalcOxidationNumbers` | no | Calvo Fracassi & Landrum 2021 |

Total: **1,015** vector/scalar + **4** oxidation-number aggregates = **1,019**.

## Design

* **Block-cache pattern (Blueprint §3.1).** Each vector block is computed with a single
  RDKit call and cached via an in-plugin `functools.lru_cache(maxsize=1)` keyed on the
  `Mol` object. All indexer descriptors of a molecule reuse that one call; the single slot
  self-evicts on the next molecule, so memory is bounded and there are never per-element
  RDKit calls.
* **Determinism.** 3D blocks depend on the conformer, which the MILIA descriptor calculator
  embeds with the pinned ETKDG seed (`randomSeed=42`). Output is identical across runs and
  process restarts.
* **NaN, never raise.** Invalid molecule, missing conformer, or an RDKit precondition
  (e.g. `GetUSR` requires ≥ 3 atoms) yields `float('nan')`; the descriptor is reported
  *skipped with reason*, not failed.
* **Oxidation aggregation.** `CalcOxidationNumbers` sets a per-atom Pauling oxidation state;
  MILIA aggregates it over **heavy atoms only** (explicit H removed), following the
  descriptor-community convention (Mordred/Todeschini strip explicit H for correctness).
  These aggregates are topological (conformer-independent).

## Naming

Every descriptor function is named identically to its descriptor `name` (e.g. the function
`WHIM_1` provides descriptor `WHIM_1`), so discovery registers each exactly once. The
`block` metadata field carries the Dragon/Todeschini block label for fine-grained provenance
without expanding the six-value category enum.

## Provenance

Counts and values verified against `rdkit==2025.03.5` (pinned runtime) and `rdkit==2026.03.5`
— identical in both. This plugin implements only published, RDKit-exposed definitions; no
proprietary parameter tables are used.
