# `cdk_substructure` — CDK Laggner substructure-count descriptors (DEP-BOUND)

First-party **DEP-BOUND, opt-in** descriptor plugin shipped with MILIA (descriptor programme,
**Pace 11 / v1.14.0**). 307 Laggner functional-group **substructure counts** plus 3 **Ghose-Crippen** descriptors (ALogP/ALogp2/AMR), computed with the
**canonical CDK engine** (`SubstructureFingerprinter.getCountFingerprint`; the Christian Laggner /
InteLigand SMARTS set) — **not** an RDKit reimplementation.

> ## ⚠ Opt-in + Java required
> This plugin needs the **`[descriptors-cdk]` extra** (`CDK-pywrapper` + `jpype1`), a **JRE**, and
> **Python ≥ 3.11** (CDK-pywrapper 1.0.0 drops Python 3.10; the dependency is marker-gated, so on 3.10 the
> extra is empty and the plugin is auto-skipped).
> Without them it is **skipped cleanly at discovery** via the plugin system's `requires_extra` gate
> (this plugin declares `requires_extra: descriptors-cdk`; the loader probes the extra and skips the
> whole plugin when it is not importable), so the base install and the default descriptor count
> (**3,026**) are unaffected — the plugin is not discovered, registered, or validated there.
> Install `pip install "milia-py[descriptors-cdk]"` (or `uv sync --extra descriptors-cdk`) and
> ensure Java is present to enable it (**→ 3,333**).

## Why an engine wrapper (not RDKit-native)

RDKit and CDK use different aromaticity/SMARTS perception. An RDKit reimplementation of the 307
Laggner SMARTS reproduces CDK on only **~98.86%** of cases (~11/307 fused-heteroaromatic/tautomer
patterns diverge — e.g. `Aromatic`, `Heteroaromatic`, `Oxoarene`, `Lactam`). Calling **CDK itself**
makes the values **canonical and bit-exact by construction**. Full rationale + the evaluation of
alternatives (padelpy vs jpype, scyjava vs CDK-pywrapper, the zero-core-edit constraint) is in the
Blueprint **§9 Pace-11 STRATEGY EVALUATION + DECISION RECORD**.

## Contents

**310 descriptors.** `SubFPC1..307` — counts of the 307 Laggner functional-group SMARTS; plus `ALogP`, `ALogp2`, `AMR` — Ghose-Crippen atomic LogP / its square / molar refractivity (CDK `ALOGPDescriptor`, `category: constitutional`, `block: ALOGP`; distinct from the base RDKit `MolLogP`/`MolMR`).

`SubFPC1..307` — counts of the 307 Laggner functional-group SMARTS (each
descriptor's Laggner label is in its `description`, e.g. `SubFPC1` = *Primary_carbon*, `SubFPC2` =
*Secondary_carbon*, …). `category: fragments`, `block: SubstructureCount`, `requires_3d: false`,
`dependencies: [CDK-pywrapper]`.

## Design (zero core modification)

* **Persistent in-plugin JVM.** One JVM is booted lazily per process (~0.4 s) via `jpype`; the
  **complete CDK jar is sourced from the `CDK-pywrapper` wheel** (offline-safe — not bundled in the
  MILIA repo, not fetched from Maven at runtime). Per-molecule compute ~7 ms; a 10k-molecule set
  ≈ 70 s (vs ~7 h for a per-call subprocess like padelpy).
* **Thread-safe.** A module lock serialises CDK calls; MILIA's descriptor calculator is sequential
  by default (`configs/descriptors.yaml parallel_computation: false`).
* **Block-cache.** One CDK call per molecule → all 307 counts cached; the 307 wrappers index it.
* **Integration point.** RDKit mol → canonical SMILES → CDK parse → counts (deterministic).
* **Purity preserved elsewhere.** This is the only plugin with a Java dependency, and it is opt-in;
  everything else in MILIA remains RDKit-native.

## Reproducibility

* Values are pinned to the CDK version shipped by the installed `CDK-pywrapper`. The frozen snapshot
  is **CDK's own integer counts** (self-consistent — engine == oracle; cross-platform reproducible).
  **Regenerate the snapshot on any `CDK-pywrapper`/CDK version bump.**
* Tests `importorskip` `CDK_pywrapper`/`jpype` and skip cleanly where the engine (or Java) is absent;
  the contract tests (counts, names, dependency declaration) run regardless.
