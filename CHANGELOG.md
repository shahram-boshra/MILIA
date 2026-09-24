# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.15.0] - 2026-09-24

### Added

- **Molecular Descriptors — Pace 11 follow-up: 3 Ghose-Crippen descriptors** (`ALogP`, `ALogp2`, `AMR`) added
  to the DEP-BOUND `cdk_substructure` plugin, computed with the canonical CDK engine (`ALOGPDescriptor`) via
  the same persistent-JVM path as the 307 Laggner substructure counts. `category: constitutional`,
  `block: ALOGP`. The plugin now provides **310** descriptors (307 + 3); with `[descriptors-cdk]` on Python
  3.11+ the total is **3,336** (default install unchanged at 3,026). Deterministic; validated against the
  CDK-generated frozen snapshot (self-consistent, engine == oracle). Ghose-Crippen ALogP/AMR are distinct from
  the RDKit `MolLogP`/`MolMR` already in the base (different atom-typed implementation).
- **Scope note:** the originally-planned edge-adjacency `EE_*` family (9) was dropped — it has no CDK
  descriptor class (it is PaDEL-specific), so it cannot use the persistent-JVM CDK engine, and it is of
  marginal value over the 177 existing `matrix_spectral` spectral descriptors.

### Fixed

- **Performance tests hardened against CI flakiness (root-cause fix).** `test_single_descriptor_speed` and
  `test_multiple_descriptors_speed` used a single absolute wall-clock measurement (`assert elapsed < 1.0/2.0`),
  a known flaky anti-pattern on shared CI (a load spike made a trivial 100×`MolWt` run take ~2.9 s). Rewritten
  to **warm up + measure best-of-5 (minimum), clearing the cache each run** so they time real compute. Per the
  Python `timeit` docs the minimum is the machine's lower bound — higher values come from other processes
  interfering, not the code — so min-of-N is immune to load spikes. Registered a `perf` marker for optional
  separate running. No production-code change; the tests remain in the gate but no longer flake.

## [1.14.1] - 2026-09-23

### Fixed

- **`[descriptors-cdk]` extra now installs `jpype1`.** The `cdk_substructure` plugin loads the CDK jar via
  `jpype` directly, but `CDK-pywrapper` does not depend on jpype (it uses `install-jdk`), so the v1.14.0 extra
  was missing `jpype1` — installing `milia-py[descriptors-cdk]` (or the CI `test-cdk` job) failed with
  `ModuleNotFoundError: No module named 'jpype'`, and the plugin was silently skipped even with the extra
  present. Added `jpype1>=1.5` (Python >= 3.11 marker, matching CDK-pywrapper) to the extra. No behavior change
  for the base install or the default descriptor count (3,026); the CDK plugin (3,333 with the extra on
  Python 3.11+ with a JRE) now activates and is gated by the `test-cdk` CI job as intended.

## [1.14.0] - 2026-09-20

### Added

- **Molecular Descriptors — Pace 11: 307 CDK Laggner substructure-count descriptors** as the first-party
  **DEP-BOUND, opt-in** `cdk_substructure` plugin (`SubFPC1..307`). Counts of the 307 Christian Laggner /
  InteLigand functional-group SMARTS, computed with the **canonical CDK engine**
  (`SubstructureFingerprinter.getCountFingerprint`) via a persistent in-process JVM (`jpype`), with the
  complete CDK jar sourced from the `CDK-pywrapper` wheel. `category: fragments`, `block: SubstructureCount`,
  `requires_3d: false`.
- **Plugin system: completed the `requires_extra` optional-dependency gate (core).** The `requires_extra`
  field existed since Pace 1 but was an inert stub; it is now wired end-to-end — a plugin that declares
  `requires_extra: <extra>` is skipped cleanly at discovery (not registered or validated) when that extra is
  not importable (probed via the plugin's own module import). This gives MILIA proper support for DEP-BOUND
  plugins: `cdk_substructure` declares `requires_extra: descriptors-cdk`, so the default install (and any env
  without the extra / Python < 3.11) keeps the count at **3,026** with the plugin cleanly absent, while an env
  with the extra sees it discovered and validated. Only plugins that declare `requires_extra` are affected;
  the other 11 plugins are untouched. Covered by a new core test (`test_requires_extra_gate`).
- **Opt-in / DEP-BOUND (first dependency-gated plugin).** Requires the new `[descriptors-cdk]` extra
  (`CDK-pywrapper` + `jpype1`), a JRE, and **Python ≥ 3.11** (CDK-pywrapper drops 3.10; marker-gated so
  MILIA core stays 3.10-compatible and the plugin auto-skips on 3.10). Without them the plugin is auto-skipped by the optional-dependency
  skip-not-fail mechanism, so the **base install and default descriptor count (3,026) are unaffected**;
  installing the extra activates it (**→ 3,333**).
- **Why an engine wrapper, not RDKit-native.** RDKit and CDK differ in aromaticity/SMARTS perception, so an
  RDKit reimplementation reproduces CDK on only ~98.86% of the 307 patterns (~11 fused-heteroaromatic/tautomer
  patterns diverge, unfixable by aromaticity-model choice). Using CDK itself makes the values canonical and
  bit-exact by construction. Full evaluation (padelpy vs jpype; scyjava vs CDK-pywrapper; the zero-core-edit
  and repo-size constraints; benchmarks) is recorded in the Blueprint §9 Pace-11 STRATEGY EVALUATION.
- **Design.** Zero core modification: the plugin owns the JVM lifecycle (lazy boot ~0.4 s, then ~7 ms/mol) and
  thread-safety (module lock; the descriptor calculator is sequential by default) entirely in-plugin;
  block-cache; `name == function_name` (0 bonus); None/failure → NaN. The CDK jar is never bundled in the repo
  (repo stays 3.8 MB) and never fetched from Maven at runtime (offline-safe).
- **Validation.** Frozen CDK snapshot (307 integer counts × panel) — self-consistent (engine == oracle),
  deterministic, cross-platform; test module (4007) covers the snapshot, determinism, contract, and NaN
  robustness, and `importorskip`s the CDK engine so CI without Java skips cleanly.

## [1.13.0] - 2026-09-19

### Added

- **Molecular Descriptors — Pace 10: 25 extended-Hückel conceptual-DFT reactivity descriptors** shipped as
  the first-party `eht_electronic` plugin (auto-discovered, zero core edits). Computed with RDKit's extended
  Hückel engine (`rdkit.Chem.rdEHTTools` = YAeHMOP) on the MILIA-core conformer
  (`AddHs → EmbedMolecule(randomSeed=42) → MMFFOptimizeMolecule`):
  - Frontier orbitals — `E_HOMO`, `E_LUMO`, `HOMO_LUMO_Gap`.
  - Conceptual DFT (Koopmans; Parr–Pearson/Gázquez) — `IonizationPotential`, `ElectronAffinity`,
    `Electronegativity`, `ChemicalPotential`, `ChemicalHardness`, `ChemicalSoftness`, `ElectrophilicityIndex`,
    `ElectrodonatingPower`, `ElectroacceptingPower`.
  - `EHT_TotalEnergy`, `EHT_FermiEnergy`; EHT partial-charge descriptors; condensed **Fukui** indices
    (f⁺/f⁻/f⁰ + dual descriptor) via finite differences on the N±1 species at fixed geometry.
  `category: electronic`, `block: EHT`, `requires_3d: true`, `dependencies: []`. Crosses **3,026** descriptors.
- **Level of theory (documented prominently).** These are **EHT-level (qualitative, semi-empirical),
  method- and geometry-dependent** reactivity descriptors — not DFT-grade; use as relative QSAR features only.
- **Validation.** RDKit-native and deterministic; frozen-conformer snapshot (MolBlock + full-precision
  coordinates + values, pinned to `rdkit==2025.3.6`) → bit-exact (300/300); plus a live re-derivation of the
  conceptual-DFT formulas from raw `rdEHTTools` primitives. Test module (401) covers the snapshot, formulas,
  determinism, robustness, contract, and analytical identities.
- **Scope decision (Blueprint §9 Pace-10 DECISION RECORD).** EHT was chosen over GFN2-xTB (`tblite`) and DFT
  after full due diligence: mordred has no quantum oracle; mainstream descriptor platforms compute no QM;
  GFN2-xTB is more accurate but is a heavy compiled dependency needing a tolerance (non-bit-exact) oracle and
  is 3–5× slower (an SCF per molecule) — architecturally misplaced in MILIA's inline, cached featurizer,
  where quantum-grade features belong in an offline/precomputed path. EHT is the canonical in-model choice
  (RDKit-native, deterministic, bit-exact, zero-dependency). Disablable via `plugins.yaml disabled_plugins`.

## [1.12.0] - 2026-09-19

### Added

- **Molecular Descriptors — Pace 9: 50 charged-partial-surface-area, gravitational & geometrical 3D
  descriptors** shipped as the first-party `cpsa_geometric_3d` plugin (auto-discovered, zero core edits).
  Computed on the MILIA-core conformer (`AddHs → EmbedMolecule(randomSeed=42) → MMFFOptimizeMolecule`):
  - **CPSA (42)** — Stanton & Jurs charged partial surface areas: `PNSA/PPSA/DPSA/FNSA/FPSA/WNSA/WPSA`
    (versions 1–5), `RNCG/RPCG`, `RNCS/RPCS`, `TASA/RASA/RPSA`. Surface area = Shrake–Rupley dot
    tessellation over a level-5 icosphere (vdW + 1.4 Å); charges = RDKit Gasteiger.
  - **GravitationalIndex (4)** — `GRAV/GRAVH/GRAVp/GRAVHp`.
  - **GeometricalIndex (4)** — `GeomDiameter/GeomRadius/GeomShapeIndex/GeomPetitjeanIndex`.
  `category: geometric`, `requires_3d: true`, `dependencies: []`.
- **Validation.** All 50 reproduce the `mordredcommunity` 2.0.7 oracle bit-exact (rel-err 0; 600/600 across
  a 12-molecule panel), verified under both rdkit 2025.3.6 (repo pin) and 2026.03.6. The frozen snapshot
  stores the exact conformer (MolBlock + full-precision coordinates) + mordred values, so the test
  reconstructs the geometry rather than re-embedding — version-robust. Test module
  `tests/test_descriptor_cpsa_geometric_3d_unit.py` (778) gates the snapshot, live mordred cross-check,
  determinism, missing-conformer/None → NaN, contract, de-dup guards, and analytical identities.
- **Design.** RDKit + NumPy + stdlib only; deterministic on the pinned conformer; `NaN` on any failure;
  compute-once-per-molecule block-cache; `name == function_name` (0 bonus discoveries). Crosses **3,000+**
  configurable descriptors (2,951 → 3,001).
- **Scope.** Pace 9 was **re-scoped** from the Blueprint's "~90, DEP-BOUND `[descriptors-3d]`": 3D ships in
  base like `addcore_3d` (RDKit-only), and after de-dup the real count is 50 — the mordred CPSA `TPSA`
  (name-clash with HAVE topological TPSA), `MomentOfInertia` (= HAVE `PMI1/2/3`) and `PBF` (= HAVE) are
  excluded.

## [1.11.0] - 2026-09-17

### Added

- **Molecular Descriptors — Pace 8: 45 Extended Topochemical Atom (ETA) descriptors** shipped as the
  first-party `eta` plugin (auto-discovered, zero core-file edits). Second-generation ETA indices of
  Roy & Ghosh: core count (`ETA_alpha`), shape (`ETA_shape_p/y/x`), valence-electron-mobile counts
  (`ETA_beta` sigma/nonsigma/delta), composite index (`ETA_eta` local/reference/functionality/branching),
  `ETA_dAlpha`, epsilon (`ETA_epsilon_1..5`), `ETA_dEpsilon`, `ETA_dBeta`, `ETA_psi`/`ETA_dPsi`, each in
  plain and averaged (`AETA_*`) forms. `category: topological`, `block: ExtendedTopochemicalAtom`.
- **Validation.** All 45 reproduce the `mordredcommunity` 2.0.7 oracle
  (`mordred.ExtendedTopochemicalAtom`) bit-exact (rel-err 0; 855/855 across a 19-molecule panel including
  aromatic heterocycles, charged, halogen, triple-bond and fused-ring cases). Test module
  `tests/test_descriptor_eta_unit.py` (985) gates the frozen snapshot + live cross-check, determinism,
  robustness, contract, and analytical guards (e.g. alkane `ETA_alpha` == 0.5·heavy-count).
- **Design.** RDKit + NumPy + stdlib only; 2D (`requires_3d: false`); deterministic; `NaN` on failure;
  disconnected or 0-atom molecules → `NaN` for all (oracle `require_connected`). Reference (all-carbon,
  single-bond) and saturated skeletons are built exactly as the oracle does. Compute-once-per-molecule
  block-cache; `name == function_name` (all valid identifiers, 0 bonus discoveries).
- **Scope.** Pace 8 was **re-scoped** from the Blueprint's original edge-adjacency + extended-rank Burden:
  mordred has neither (its `BCUT` is only `1h/1l`, already shipped in `matrix_spectral`), so those
  Dragon-specific families carry no mordred oracle and were **deferred to Pace 11** (PaDEL/CDK-residual).
  ETA is the largest clean, mordred-backed, zero-collision 2D family that remained.

## [1.10.0] - 2026-09-17

### Added

- **Molecular Descriptors — Pace 7: 466 atom-typing descriptors** shipped as two first-party
  `PLUGIN-NATIVE` descriptor plugins, auto-discovered with zero core-file edits:
  - **`estate_atomtype` (316)** — Kier-Hall electrotopological-state (E-State) atom-type descriptors:
    `N`/`S`/`MAX`/`MIN` (count, sum, max, min of E-State values) over 79 Kier-Hall atom types.
    Computed RDKit-natively (`rdkit.Chem.EState` `TypeAtoms` + `EStateIndices`, H-suppressed graph),
    `category: electronic`, `block: AtomTypeEState`.
  - **`cats2d` (150)** — CATS2D topological pharmacophore-pair descriptors (Chemically Advanced
    Template Search; Schneider et al. 1999): 15 canonical potential-pharmacophore-point pairs
    (D/A/P/N/L) × 10 topological-distance bins (0-9), raw counts. `category: fragments`,
    `block: CATS2D`.
- **Validation.** All 466 reproduce an independent oracle bit-exact: E-State vs `mordredcommunity`
  2.0.7 (3160/3160, rel-err 0; RDKit's 79-label vocabulary proven identical to the oracle), CATS2D vs
  the peer-reviewed PyBioMed reference implementation (Dong et al. 2018) at `scale=1` raw counts
  (2100/2100, 0 mismatches). Test modules `tests/test_descriptor_estate_atomtype_unit.py` (5082) and
  `tests/test_descriptor_cats2d_unit.py` (2715) gate frozen oracle snapshots, determinism, robustness,
  contract (registration counts, name==function_name, no HAVE/prior-pace collision), and assertive
  guards.
- **Design.** Both families 2D (`requires_3d: false`), pure (RDKit + stdlib), deterministic, `NaN` on
  failure; 0-atom (invalid) molecule → `NaN` for all. `name == function_name` (all valid identifiers,
  0 bonus discoveries). Compute-once-per-molecule block-cache (`functools.lru_cache(maxsize=1)`).
- **Scope.** `MoeType` (53) was **excluded** — all 53 names (`LabuteASA`, `PEOE_VSA*`, `SMR_VSA*`,
  `SlogP_VSA*`, `EState_VSA*`, `VSA_EState*`) are already in the HAVE RDKit baseline (name-uniqueness
  invariant). The Ghose-Crippen `ALOGP`/`ALOGP2`/`AMR` triplet was **deferred to Pace 11**
  (PaDEL/CDK residual): oracle identified and verified (PaDEL 2.21 via `padelpy`; distinct from HAVE
  `MolLogP`/`MolMR`), but faithful native reproduction requires porting CDK's incomplete 120-type
  atom-typer — disproportionate for 3 descriptors and squarely a CDK-residual task.

## [1.9.0] - 2026-09-17

### Added

- **Molecular Descriptors — Pace 6: 606 2D topological autocorrelation descriptors** shipped as the
  first-party `autocorrelation_2d` descriptor plugin (`plugins/descriptors/autocorrelation_2d/`),
  implemented `PLUGIN-NATIVE` from primary literature and auto-discovered with zero core-file edits.
  Families (all H-included graph, `explicit_hydrogens=True`):
  - **ATS** — Moreau-Broto autocorrelation w^T B_k w (Moreau & Broto 1980), 11 props × lags 0-8 (99)
  - **AATS** — Averaged Moreau-Broto ATS_k / Δ_k, 11 props × lags 0-8 (99)
  - **ATSC** — Centred Moreau-Broto, 12 props × lags 0-8 (108; Todeschini 2009)
  - **AATSC** — Averaged + centred Moreau-Broto, 12 props × lags 0-8 (108)
  - **MATS** — Moran coefficient N·AATSC_k/var(w_c), 12 props × lags 1-8 (96; Moran 1950)
  - **GATS** — Geary coefficient, 12 props × lags 1-8 (96; Geary 1954)
- **Validation.** All 606 reproduce the `mordredcommunity` oracle at `rtol ≤ 1e-4` (actual ≤ 6e-7).
  Test module `tests/test_descriptor_autocorrelation_2d_unit.py` gates a frozen oracle snapshot (with
  a live `mordredcommunity` cross-check when installed), determinism, robustness, contract (606
  registrations, family composition), and assertive guards (AATS0 denominator regression, MATS/GATS
  lag=0 engine guard, zero HAVE/addcore_3d name collision).
- **Design.** All 2D (`requires_3d: false`), pure (RDKit + stdlib + numpy), deterministic, `NaN` on
  failure. `name == function_name` (all valid Python identifiers, 0 bonus discoveries). Single
  `_autocorr(mol, kind, lag, short)` engine shared across all 6 families.
- **Scope.** Zero de-dup required: mordred 2D autocorrelations (named `ATS*/MATS*/GATS*`) are
  entirely distinct from `addcore_3d`'s `AUTOCORR3D_*` (3D, conformer-based) and RDKit's
  `AUTOCORR2D_*` (different weighting). **AATS0 normalization bug** (Δ_0 = n_atoms, not ½n)
  was caught during the build and is regression-guarded in the test suite.

## [1.8.0] - 2026-09-15

### Added

- **Molecular Descriptors — Pace 5: 177 eigenvalue/matrix-spectral descriptors** shipped as the
  first-party `matrix_spectral` descriptor plugin (`plugins/descriptors/matrix_spectral/`), implemented
  `PLUGIN-NATIVE` from primary literature and auto-discovered with zero core-file edits. Families:
  - **AdjacencyMatrix** — `SpAbs/SpMax/SpDiam/SpAD/SpMAD/LogEE/VE1-3/VR1-3` (12)
  - **DistanceMatrix** — same spectra (12)
  - **DetourMatrix** — + `SM1` (13; Trinajstić)
  - **BaryszMatrix** — spectra × 8 atomic-property weightings `Z/m/v/se/pe/are/p/i` (104; Barysz et al. 1983)
  - **BCUT** — Burden eigenvalues × 12 weightings × hi/lo (24; Pearlman & Smith 1999)
  - **MolecularId** — `MID/AMID` × {any, hetero, C, N, O, X} (12; Randić 1984)
- **Validation.** All 177 reproduce the `mordredcommunity` oracle to `rtol ≤ 1e-4` (in practice ≤ 6e-8).
  Test module `tests/test_descriptor_matrix_spectral_unit.py` gates a frozen oracle snapshot (with a live
  cross-check when installed), determinism, robustness (invalid / disconnected → `NaN`), contract (177
  unique registrations, per-family composition, BCUT name-sanitization), de-dup guards, and SM1-trace
  boundary identities.
- **Design.** All 2D (`requires_3d: false`), pure (RDKit + stdlib + numpy, no networkx), deterministic;
  `require_connected` → `NaN` on disconnected graphs, never raises. `name == function_name`; the 24 BCUT
  names are hyphen-sanitized to valid identifiers (canonical Mordred string in each description) → 0 bonus.
- **Scope.** mordred BCUT is kept (different Burden convention than the HAVE RDKit `BCUT2D`, not a
  duplicate); the scalar `DetourIndex` is not shipped here (it is in `walk_path_information`, Pace 4).

## [1.7.0] - 2026-09-13

### Added

- **Molecular Descriptors — Pace 4: 86 walk/path + information-content descriptors** shipped as the
  first-party `walk_path_information` descriptor plugin (`plugins/descriptors/walk_path_information/`),
  implemented `PLUGIN-NATIVE` from primary literature and auto-discovered with zero core-file edits.
  Families:
  - **WalkCount** — `MWC01-10`, `TMWC10`, `SRW02-10`, `TSRW10` (Rücker & Rücker 1993; Harary 1969)
  - **PathCount** — `MPC2-10`, `TMPC10`, `piPC1-10`, `TpiPC10` (Randić 1979)
  - **DetourIndex** — `DetourIndex` (Trinajstić, *Chemical Graph Theory*)
  - **InformationContent** — `IC/TIC/SIC/BIC/CIC/MIC/ZMIC` × orders 0-5 (Bonchev & Trinajstić 1977; Basak 1999)
  - **VertexAdjacencyInformation** — `VAdjMat` (Todeschini & Consonni, *Handbook*)
- **Validation.** All 86 reproduce the `mordredcommunity` oracle **bit-exact** (`rtol ≤ 1e-4`). Test module
  `tests/test_descriptor_walk_path_information_unit.py` gates a frozen oracle snapshot (with a live
  `mordredcommunity` cross-check when installed), determinism, robustness, and contract (86 unique
  registrations, per-family composition, boundary values).
- **Design.** All 2D (`requires_3d: false`), pure (RDKit + stdlib + numpy), deterministic, `NaN` on invalid
  input (never raises); `name == function_name` (all valid identifiers → 0 bonus discoveries). WalkCount /
  PathCount / DetourIndex / VAdjMat use the heavy-atom graph; InformationContent uses the H-included,
  kekulized graph (the oracle's default). The detour index (longest-simple-path, NP-hard) has a DFS
  step-budget that returns `NaN` for pathological fused-polycyclics rather than hanging.
- **Scope.** `MWC03 = 2·Zagreb2` is a known affine correlation (distinct value, not a duplicate) — both
  shipped. The 13 detour-matrix eigenvalue-spectra (`SpAbs_Dt`/`VE*_Dt`/`VR*_Dt`/`SM1_Dt`/`LogEE_Dt`) are
  deferred to the eigenvalue/matrix-spectra pace (Pace 5).

## [1.6.0] - 2026-09-12

### Added

- **Molecular Descriptors — Pace 3: 107 topological/connectivity indices** shipped as the first-party
  `topological_connectivity` descriptor plugin (`plugins/descriptors/topological_connectivity/`),
  implemented `PLUGIN-NATIVE` from primary literature and auto-discovered with zero core-file edits.
  Families:
  - **Wiener** — `WPath`, `WPol` (Wiener 1947)
  - **Zagreb** — `Zagreb1`, `Zagreb2`, `mZagreb1`, `mZagreb2` (Gutman & Trinajstić 1972)
  - **Atom-bond connectivity** — `ABC`, `ABCGG` (Estrada 1998; Graovac-Ghorbani 2016)
  - **Eccentric connectivity** — `ECIndex` (Sharma, Goswami & Madan 1997)
  - **Topological shape** — `Diameter`, `Radius`, `TopoShapeIndex`, `PetitjeanIndex` (Petitjean 1992)
  - **Schultz** — `MTI`, `SchultzIndex`, `ModSchultzIndex` (Schultz 1989)
  - **Gálvez topological charge** — `GGI1-10`, `JGI1-10`, `JGT10` (Gálvez et al. 1994)
  - **Molecular distance edge** — `MDEC`/`MDEO`/`MDEN` (Liu, Cai & Yao 1998)
  - **Kier-Hall Chi** — path/cluster/path-cluster/chain, simple + valence (Kier & Hall 1976/1986)
- **Validation.** The 104 Mordred-defined descriptors reproduce the `mordredcommunity` oracle **bit-exact**
  (`rtol ≤ 1e-4`); the 3 Schultz indices are gated against published reference values (benzene 132/54/54;
  propane 5/3) and the identity `MTI = Zagreb1 + 2·SchultzIndex`. Test module
  `tests/test_descriptor_topological_connectivity_unit.py` gates a frozen oracle snapshot (with a live
  `mordredcommunity` cross-check when installed), determinism, robustness, and contract (107 unique
  registrations).
- **Design.** All are 2D (`requires_3d: false`), pure (RDKit + stdlib + numpy), deterministic, and return
  `NaN` (never raise) on invalid input. Registered `name` equals `function_name` (valid identifiers, e.g.
  `Xp_0d`, `MDEC_11`; the canonical Mordred string is preserved in each descriptor's `description`) —
  honouring the single-registration invariant, so all 107 register exactly once with 0 bonus discoveries.
- **Scope.** The 5 valence-path low orders `Xp-0dv…Xp-4dv` are de-duplicated (not shipped — identical to
  the existing `HAVE` RDKit `Chi0v…Chi4v`). The ~166 eigenvalue/matrix-spectra descriptors
  (Barysz/adjacency/distance/detour spectra, BCUT) are deferred to a dedicated later pace.

## [1.5.0] - 2026-09-11

### Added

- **Molecular Descriptors — Pace 2: 20 constitutional-ext + property/physicochemical descriptors** shipped
  as the first-party `constitutional_property` descriptor plugin
  (`plugins/descriptors/constitutional_property/`), implemented `PLUGIN-NATIVE` from primary literature and
  auto-discovered with zero core-file edits. Descriptors:
  - **McGowan volume** — `VMcGowan` (Abraham & McGowan 1987)
  - **vdW volume (atom-and-bond)** — `Vabc` (Zhao, Abraham & Zissimos 2003)
  - **Polarizability** — `apol`, `bpol` (Miller 1990)
  - **CarbonTypes** — `C1SP1 C2SP1 C1SP2 C2SP2 C3SP2 C1SP3 C2SP3 C3SP3 C4SP3` + `HybRatio`
    (Todeschini & Consonni 2009; Yang et al. 2010)
  - **Framework / complexity** — `fMF` (Bemis & Murcko 1996), `fragCpx` (Nilakantan et al. 2006)
  - **Drug-likeness rule filters** — `Lipinski`, `GhoseFilter`, `VeberFilter`, `EganFilter`
    (Lipinski 2001; Ghose 1999; Veber 2002; Egan 2000)
- **Validation.** The 18 Mordred-defined descriptors reproduce the `mordredcommunity` oracle to
  `rtol ≤ 1e-4` (bit-identical on the validation panel); `VeberFilter`/`EganFilter` are gated against their
  published thresholds. Test module `tests/test_descriptor_constitutional_property_unit.py` gates a frozen
  oracle snapshot (with a live `mordredcommunity` cross-check when installed), determinism, robustness, and
  contract (20 unique registrations, `block`/`category`/`requires_3d` metadata).
- **Design.** All are 2D (`requires_3d: false`), pure (RDKit + stdlib only), deterministic, and return
  `NaN` (never raise) on invalid input or an out-of-parameter element; boolean rule flags honour the scalar
  contract by returning `1.0`/`0.0`.
- **Scope.** `FCSP3` is not shipped (numerically identical to the existing `HAVE` `FractionCSP3`,
  de-duplicated); MLFER/Abraham descriptors are deferred to the PaDEL/CDK-residual pace (no RDKit/mordred
  oracle).

## [1.4.0] - 2026-09-02

### Added

- **Molecular Descriptors — Pace 1: 1,019 ADD-CORE RDKit 3D descriptors** shipped as the first-party `addcore_3d` descriptor plugin (`plugins/descriptors/addcore_3d/`), extending the descriptor universe beyond the ~492-descriptor RDKit `HAVE` baseline. Blocks (RDKit-native, verified against the pinned runtime): WHIM (114), GETAWAY (273), RDF (210), 3D-MoRSE (224), Autocorr3D (80), USR (12), USRCAT (60), MQN (42), and heavy-atom oxidation-number aggregates (min/max/mean/sum). Vector blocks use a compute-once-per-molecule block-cache (`functools.lru_cache(maxsize=1)` keyed on the molecule), exposing each element as an individually-named scalar descriptor while issuing a single RDKit call per block. All descriptors are deterministic under the pinned ETKDG seed (42) and return `NaN` (never raise) on invalid input or unmet preconditions (e.g. USR's ≥3-atom requirement). One-per-family test module `tests/test_descriptor_addcore3d_unit.py` gates ground-truth-vs-RDKit-oracle equality, determinism, robustness, and contract (1,019 unique registrations, `block`/`category`/`requires_3d` metadata).
- Descriptor-plugin discovery is now config-driven for the descriptor kind-container: `configs/descriptors.yaml` gains a `molecular_descriptors.plugins` block (`enabled`, `plugin_paths`, `auto_discover`, `auto_validate`) pointing at `milia_pipeline/plugins/descriptors/` — adding a descriptor plugin requires no core-file edits.

### Changed

- `DescriptorMetadata` (`descriptors/descriptor_categories.py`) gains a non-breaking optional `block` field carrying the Dragon/Todeschini block-provenance label (e.g. `WHIM`, `GETAWAY`), so descriptors record fine-grained provenance without expanding the six-value `DescriptorCategory` enum per release. `DescriptorDeclaration` (`descriptors/descriptor_plugin_system.py`) gains matching optional `block` and `requires_extra` fields (the latter reserved for optional-dependency skip-not-fail wiring in later paces); both are threaded from `plugin.yaml` through registration into the registry metadata. All additions default to `None`/absent — existing plugins and metadata are unaffected.
- Pinned `rdkit==2025.3.5` → `rdkit==2025.3.6`. Release 2025.03.6 fixes GETAWAY nondeterminism ([rdkit#7264](https://github.com/rdkit/rdkit/issues/7264)); the previous pin was the last release exhibiting call-to-call GETAWAY variance, which would otherwise violate the descriptor determinism gate. Baseline descriptor values are bit-identical across the two patch releases (no regression).


## [1.3.0] - 2026-08-26

### Added

- **55 novel graph transforms across 9 first-party plugin families**, all configuration-driven and composable: graph products (Cartesian, Tensor, Strong, Lexicographic, Rooted, Corona), combinatorial & dual operators (line/Levi graphs, quotient, transitive closure/reduction, bipartite projection, graph dual), curvature-based rewiring (Forman–Ricci, effective-resistance, spectral-gap, resistance-curvature), topological lifts (hypergraph clique/star expansion, persistent-homology features), spectral & signal transforms, graph reduction/coarsening, structural padding for static batching, non-spatial augmentation, and latent-structural encodings (motif compression, structural-role, anonymous-walk). Brings the discoverable transform total to 130+.

### Changed

- Reorganised `milia_pipeline/plugins/` into per-kind containers: transformation plugins now live under `plugins/transformations/` (symmetric with `plugins/descriptors/`), and a `plugins/models/` container was added. Introduced `user_template/` scaffolds for the transformation and model kinds and `get_transform_plugins_directory()` / `get_model_plugins_directory()` helpers; model plugin discovery now resolves package-relative (CWD-independent) instead of relative to the working directory. The `pyg_augmentation` plugin moved into `plugins/transformations/`; the empty `plugins/myplugins/` scaffold was removed (users add plugins via each kind's `user_template/` or an external `plugins.plugin_paths` entry).
- Made the plugin compatibility smoke-test precondition-aware: a new `TransformPreconditionError` distinguishes transforms that require caller-supplied inputs (a second operand graph, a partition, or a graph property such as planarity/acyclicity/bipartiteness) from genuine defects. Such transforms are now reported as skipped (informational) rather than failed; constructor parameters are synthesised from declared constraints where possible, and paramless-constructor defects are still surfaced as failures.

### Fixed

- `RandomNodeSample` now defaults to `ratio=0.5` when neither `num` nor `ratio` is given, making it usable bare and consistent with the sibling augmentation transforms (which default `p=0.5`); with `num` set, `ratio` remains `None`.
- Logging setup tests: removed a stale `inspect` mock (the code no longer uses `inspect`) and corrected an obsolete file-handler-`OSError` expectation — `setup_logging` intentionally falls back to console-only logging on file errors rather than raising. Aligned the `Raises:` docstring accordingly.

## [1.2.2] - 2026-08-08

### Added

- Bundled sample inputs under `test_data/` (`molecules.csv`, `molecules_inchi.csv`, `molecules.sdf`, `molecules.xyz`) covering every `--predict` input format; re-tracked via `.gitignore` (text fixtures only — heavy binaries remain ignored, and the samples stay excluded from the PyPI sdist/wheel).

### Changed

- `--predict` walkthrough in `README.md`, `QUICKSTART.md`, and `docs/getting-started.md` now points at the bundled `test_data/molecules.csv`.
- Enabled the models master switch in `configs/models.yaml`.

### Fixed

- Corrected stale documentation: checkpoint filename `best_model.pt` → `best.pt`; test-suite counts `127` → 163 test files / 1,965 smoke; removed the inaccurate "pytest runs inside the container" claim (the runtime image has no test tooling).

## [1.2.1] - 2026-08-06

### Changed

- PyPI distribution name is now `milia-py` (installed via `pip install milia-py`).
  The bare name `milia` was unavailable on PyPI (blocked by its name-similarity
  check), so the distribution name is decoupled from the import name — a standard
  practice (cf. `scikit-learn`/`sklearn`, `Pillow`/`PIL`). The import package
  (`import milia_pipeline`), the `milia` CLI entry point, the GitHub repository
  (`MILIA`), and the Zenodo/Mendeley DOIs are unchanged.

### Added

- First automated PyPI release via GitHub Actions **Trusted Publishing** (OIDC,
  no API tokens), gated through a `pypi` deployment environment.
- `PyPI` project URL in `[project.urls]`.

## [1.2.0] - 2026-06-26

### Added

- QM40 dataset support: 162,954 neutral drug-like ZINC molecules (10-40 heavy
  atoms) at B3LYP/6-31G(2df,p), with optimized geometries, Mulliken charges,
  16 scalar quantum-mechanical properties, and per-bond local vibrational mode
  force constants. Reference: Madushanka, Moura Jr. & Kraka, *Scientific Data*
  11, 1376 (2024). Includes dataset implementation, handler, preprocessor (ZIP
  of three CSV files joined by Zinc_id), CSV parser, colocated YAML
  configuration, and `config_constants` registry entries. Uses the
  `coordinate_based` molecule-creation strategy (SMILES present, no InChI) and
  is neutral-only (`supports_charged_molecules()` is `False`).

## [1.1.0] - 2026-02-12

### Added

- Multi-format molecular conversion with RDKit integration (`molecules/`).
- Automated structural and chemical feature extraction and enrichment.
- PyTorch Geometric compatible dataset implementation (`miliaDataset`).
- Modular wavefunction data preprocessing for MOLDEN and FCHK formats.
- Extensible graph transformation system with experimental setup support.
- Three-tier plugin architecture for descriptors, transformations, and general extensions.
- Unified dataset handler pattern with DFT and DMC support (`create_handler`).
- Schema-validated YAML configuration system (`config/`).
- Comprehensive CLI with interactive mode and `milia` entry point.
- GNN model training with hyperparameter optimization support (`models/`).
- Post-training prediction and inference workflow with checkpoint support (`models/post_training/`).
- Transfer learning via `FineTuner` and `FreezeStrategy`.
- Multi-format input support via `DataConverterRegistry`.
- Molecular descriptor calculation with plugin system (`descriptors/`).
- Three-tier exception hierarchy with 50+ specialized exception classes.
- Registry integration for dataset type validation and CLI diagnostics.
- Full test suite with pytest configuration (`tests/`).
- MIT license.
- Production `pyproject.toml` with PEP 517/518/621/639 compliance.
- Comprehensive `README.md` with installation, quick start, and API reference.

[unreleased]: https://github.com/shahram-boshra/MILIA/compare/v1.9.0...HEAD
[1.9.0]: https://github.com/shahram-boshra/MILIA/compare/v1.8.0...v1.9.0
[1.8.0]: https://github.com/shahram-boshra/MILIA/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/shahram-boshra/MILIA/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/shahram-boshra/MILIA/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/shahram-boshra/MILIA/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/shahram-boshra/MILIA/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/shahram-boshra/MILIA/compare/v1.2.2...v1.3.0
[1.2.2]: https://github.com/shahram-boshra/MILIA/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/shahram-boshra/MILIA/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/shahram-boshra/MILIA/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/shahram-boshra/MILIA/releases/tag/v1.1.0
