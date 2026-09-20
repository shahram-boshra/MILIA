# `eht_electronic` — extended-Hückel conceptual-DFT reactivity descriptors

First-party descriptor plugin shipped with MILIA (descriptor programme, **Pace 10 / v1.13.0**).
25 `PLUGIN-NATIVE` quantum-chemical **reactivity** descriptors computed with RDKit's extended
Hückel engine (`rdkit.Chem.rdEHTTools` = YAeHMOP), deterministic and validated bit-exact against
a frozen-conformer snapshot.

> ## ⚠ Read first — level of theory
> These are **extended-Hückel-level (qualitative, semi-empirical) reactivity descriptors — NOT
> DFT-grade.** Extended Hückel theory (Hoffmann, 1963) captures interpretable frontier-orbital
> *trends*, not physically-accurate absolute energies. **Every value is method- and
> geometry-dependent**: it is defined only within this fixed protocol (an EHT single point on the
> MMFF-optimised, seed-42 ETKDG conformer) and is **not comparable** to HOMO/LUMO, χ, ω, … from a
> different method (GFN2-xTB, DFT) or a different geometry. Use them as **relative QSAR/QSPR
> features**, never as physical observables. The rationale for choosing EHT over GFN2-xTB/DFT for
> this in-pipeline, deterministic, dependency-free slot is recorded in the Blueprint §9 Pace-10
> **DECISION RECORD**.

## Contents (25 descriptors)

| Group | Descriptors |
|---|---|
| Frontier orbitals | `E_HOMO`, `E_LUMO`, `HOMO_LUMO_Gap` |
| Conceptual DFT (Koopmans; Parr–Pearson/Gázquez) | `IonizationPotential`, `ElectronAffinity`, `Electronegativity`, `ChemicalPotential`, `ChemicalHardness`, `ChemicalSoftness`, `ElectrophilicityIndex`, `ElectrodonatingPower`, `ElectroacceptingPower` |
| EHT energies | `EHT_TotalEnergy`, `EHT_FermiEnergy` |
| EHT charges | `Max/Min/MaxAbs/MinAbs/SumAbs/Mean EHTCharge` |
| Condensed Fukui (finite-difference, fixed geometry) | `MaxFukuiNucleophilic` (f⁺), `MaxFukuiElectrophilic` (f⁻), `MaxFukuiRadical` (f⁰), `Max/MinDualDescriptor` |

`category: electronic`, `block: EHT`, `requires_3d: true`, `dependencies: []`.

## Design

* **RDKit-native.** All descriptors derive from `rdEHTTools.RunMol` on the MILIA-core conformer
  (`AddHs → EmbedMolecule(randomSeed=42) → MMFFOptimizeMolecule`). No external QM engine, no extra
  dependency.
* **Conceptual DFT via Koopmans:** IP = −E_HOMO, EA = −E_LUMO; χ = (IP+EA)/2, η = (IP−EA)/2,
  ω = μ²/(2η); electrodonating/accepting powers per Gázquez et al.
* **Condensed Fukui:** finite differences of EHT atomic charges of the N, N−1 (cation) and N+1
  (anion) species at the **same geometry** (Yang–Mortier); the net charge is set as a formal charge
  used only for the electron count (verified placement-invariant), aggregated to molecular max/min.
* **Block-cache.** One memoised `_eht_block(mol)` runs the three EHT single-points once; the 25
  wrappers index it.
* **Determinism & purity.** RDKit + NumPy + stdlib only; thread-invariant; `name == function_name`
  (0 bonus); missing conformer / None / failure → NaN.

## Reproducibility & cost

* **Version-pinned.** `rdEHTTools` is marked *experimental (may change between RDKit releases)*, so
  the frozen snapshot (MolBlock + full-precision coordinates + values) is pinned to `rdkit==2025.3.6`
  (the repo pin). **Regenerate the snapshot on any RDKit upgrade.**
* **Cost tier & opt-out.** EHT is a single-point in the same conformer-cost tier as the other 3D
  plugins (`cpsa_geometric_3d`, `addcore_3d`); Fukui adds two more single-points. For
  latency-sensitive deployments, disable the whole plugin via `configs/plugins.yaml` →
  `disabled_plugins: ["eht_electronic"]`.

## Scoping notes

* **Why EHT, not GFN2-xTB or DFT?** GFN2-xTB (`tblite`) is more physically grounded but is a heavy
  compiled dependency, needs a tolerance-based (non-bit-exact) oracle, and — running an SCF per
  molecule — is 3–5× the EHT cost and architecturally misplaced in an inline, cached featurizer.
  DFT is non-deterministic across platforms. EHT is the canonical in-model choice (RDKit-native,
  deterministic, bit-exact, zero-dependency). Full analysis: Blueprint §9 Pace-10 DECISION RECORD.
* **No collision.** `E_HOMO`/`Electronegativity`/`MaxFukui*`/… are disjoint from HAVE and every prior
  pace; these are genuinely new (RDKit's `Descriptors` computes none of them).
