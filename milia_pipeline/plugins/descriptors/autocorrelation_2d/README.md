# `autocorrelation_2d` — 2D topological autocorrelation descriptors

First-party descriptor plugin shipped with MILIA (descriptor programme, **Pace 6 / v1.9.0**).
606 `PLUGIN-NATIVE` 2D autocorrelation descriptors implemented from primary literature and
validated against `mordredcommunity` (rel-err ≤ 1e-4; actual ≤ 6e-7, the residual being
floating-point accumulation on sums of vdW volumes).

## Contents (606 descriptors)

| Block | Family | Formula | Lags | Props | Count |
|---|---|---|---|---:|---:|
| `BrotoMoreau` | `ATS` | Σ w²ᵢ (k=0), ½ wᵀBₖw | 0–8 | 11 | 99 |
| `BrotoMoreau` | `AATS` | ATS_k / Δ_k | 0–8 | 11 | 99 |
| `BrotoMoreau` | `ATSC` | Centred Moreau-Broto | 0–8 | 12 | 108 |
| `BrotoMoreau` | `AATSC` | Averaged + centred | 0–8 | 12 | 108 |
| `MoranGeary` | `MATS` | N·AATSC_k / var(w_c) | 1–8 | 12 | 96 |
| `MoranGeary` | `GATS` | Geary coefficient | 1–8 | 12 | 96 |

Property weightings (11 for ATS/AATS, 12 for ATSC/AATSC/MATS/GATS — adds `c`):
`Z m v se pe are p i d dv` [+ `c` (Gasteiger charge)]

Total: **606** (2D; `requires_3d: false`; H-included graph `explicit_hydrogens=True`).

## Design

* **Native from literature.** Single `_autocorr(mol, kind, lag, short)` engine: computes
  the H-included property vector `w` (static tables for `m/v/se/pe/are/p/i`; RDKit +
  Kier-Hall formulas for `d/dv/s/c`), the lag-k indicator matrix `G = (D==k)`, and the
  family-specific formula. No Mordred calls at runtime.
* **Oracle-gated.** All 606 reproduce `mordredcommunity` at `rtol ≤ 1e-4`.
* **Purity.** RDKit + stdlib + numpy only; `name == function_name` (all valid identifiers;
  no sanitization needed); all 606 registered → 0 bonus discoveries.

## Scoping notes

* **Zero de-dup required.** mordred's 2D autocorrelations (`ATS*/MATS*/GATS*`) are distinct
  from RDKit `AUTOCORR2D_*` (different weighting/normalization) and from `addcore_3d`'s
  `AUTOCORR3D_*` (3D, conformer-based). No name collision with HAVE or any prior pace.
* **AATS0 normalization.** Δ_0 = n (atom count), **not** ½·n — regression-guarded in the
  test suite (the 2× bug was caught and fixed during the build).
* **MATS/GATS**: defined only for lag ≥ 1 (correlation between *pairs*); lag=0 → NaN.
