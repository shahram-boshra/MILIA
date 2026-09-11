# `constitutional_property` — constitutional-ext + property/physicochemical descriptors

First-party descriptor plugin shipped with MILIA (descriptor programme, **Pace 2 / v1.5.0**).
It implements 20 `PLUGIN-NATIVE` descriptors from primary literature under the standard
zero-core-modification plugin contract, and validates each against an oracle
(`mordredcommunity` for the 18 Mordred-defined descriptors; the published rule definitions
for the two absorption filters).

## Contents (20 descriptors)

| Block | Descriptors | Count | Category | Primary reference |
|---|---|---:|---|---|
| `McGowanVolume` | `VMcGowan` | 1 | constitutional | Abraham & McGowan 1987 |
| `VdwVolumeABC` | `Vabc` | 1 | constitutional | Zhao, Abraham & Zissimos 2003 |
| `Polarizability` | `apol`, `bpol` | 2 | constitutional | Miller 1990 |
| `CarbonTypes` | `C1SP1 C2SP1 C1SP2 C2SP2 C3SP2 C1SP3 C2SP3 C3SP3 C4SP3 HybRatio` | 10 | constitutional | Todeschini & Consonni 2009; Yang et al. 2010 |
| `Framework` | `fMF` | 1 | constitutional | Bemis & Murcko 1996 |
| `FragmentComplexity` | `fragCpx` | 1 | constitutional | Nilakantan et al. 2006 |
| `Lipinski` | `Lipinski`, `GhoseFilter` | 2 | drug_likeness | Lipinski et al. 2001; Ghose et al. 1999 |
| `DrugLikenessFilter` | `VeberFilter`, `EganFilter` | 2 | drug_likeness | Veber et al. 2002; Egan et al. 2000 |

Total: **20** descriptors (all 2D — `requires_3d: false`, no optional extra).

## Design

* **Native from literature (Blueprint §1, §3).** Each descriptor is implemented from its
  primary source using the published atomic constants (McGowan volumes, Bondi radii,
  atomic polarizabilities). No Dragon/proprietary code.
* **Oracle-gated (Blueprint §6).** The 18 Mordred-defined descriptors reproduce the
  `mordredcommunity` oracle to `rtol<=1e-4` (bit-identical on the validation panel);
  `VeberFilter`/`EganFilter` are validated against their published thresholds.
* **Purity (Blueprint §4).** Imports only RDKit + stdlib. Deterministic (2D, no RNG).
  NaN on any invalid molecule / undefined value — never raises. Boolean rule flags honour
  the scalar `-> float` contract by returning `1.0`/`0.0`.
* **Single registration.** `function_name == name` for every descriptor, so the plugin
  undeclared-scan registers each exactly once.

## Hydrogen convention

`VMcGowan`, `Vabc`, `apol`, `bpol` and `fMF` operate on the H-included graph (the oracle's
default); `CarbonTypes` and `fragCpx` operate on the heavy-atom graph. These match the
oracle exactly.

## Scoping notes

* `FCSP3` (Mordred CarbonTypes) is **not shipped**: it is numerically identical to the
  existing `FractionCSP3` in the MILIA `HAVE` baseline (verified bit-identical), so it is
  de-duplicated per the exhaustive-list union methodology. `HybRatio` (a distinct ratio
  that excludes sp-carbons from the denominator) **is** shipped.
* `MLFER` (Abraham E/S/A/B/V/L) is deferred to the PaDEL/CDK-residual pace: it is absent
  from both RDKit and `mordredcommunity`, so it has no in-toolkit oracle and is out of
  scope for this low-risk pace.
