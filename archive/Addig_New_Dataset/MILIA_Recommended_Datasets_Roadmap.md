# MILIA Pipeline: Recommended Datasets Roadmap

**Version:** 1.0.0 
**Created:** 2025-12-27 
**Purpose:** Prioritized roadmap for adding new datasets to MILIA to achieve competitive differentiation in the molecular ML software market 
**Evidence-Based:** All recommendations derived from web searches of current literature, benchmarks, and community adoption

---

## Executive Summary

This document provides a prioritized list of **25+ datasets** recommended for integration into the MILIA pipeline. The datasets are organized into four tiers based on their strategic importance for competitive positioning in the molecular machine learning market.

### Coverage Areas

| Domain | Datasets | Market Impact |
|--------|----------|---------------|
| Neural Network Potentials | ANI-1x, ANI-1ccx, ANI-2x, MD17/rMD17 | Foundation for universal force fields |
| Drug Discovery | QMugs, SPICE, Tox21, ToxCast, ADMET suite | Pharmaceutical industry adoption |
| Materials Science | OC20, OC22, OMat24 | Energy & catalysis applications |
| Conformer Generation | GEOM, GEOM-Drugs | 3D molecular modeling |
| Benchmarking | MoleculeNet suite | Community comparison standards |

---

## Tier 1: ESSENTIAL (Must-Have for Competitive Positioning)

These datasets are **critical** for MILIA to be taken seriously as a comprehensive molecular ML platform.

---

### 1. ANI-1x

| Property | Value |
|----------|-------|
| **Priority** | 🔴 HIGHEST |
| **Size** | ~5 million DFT conformations |
| **Molecules** | ~57,000 molecules (H, C, N, O) |
| **Level of Theory** | ωB97X/6-31G(d) DFT |
| **Properties** | Energies, forces |
| **Source** | Active learning from GDB-11, ChEMBL |

**Why Essential:**
- Gold standard for neural network potential training
- Active learning sampled for maximum diversity
- Foundation dataset for ANI-2x
- Most cited NNP training dataset in literature

**Reference:**
> Smith, J.S. et al. "The ANI-1ccx and ANI-1x data sets, coupled-cluster and density functional theory properties for molecules." Scientific Data 7, 134 (2020).

**Implementation Notes:**
- Contains SMILES and coordinates
- NPZ format available
- `identifier_keys`: InChI first recommended

---

### 2. ANI-1ccx

| Property | Value |
|----------|-------|
| **Priority** | 🔴 HIGHEST |
| **Size** | 500,000 data points |
| **Molecules** | Subset of ANI-1x |
| **Level of Theory** | CCSD(T)/CBS extrapolation |
| **Properties** | Energies (coupled-cluster accuracy) |
| **Source** | Intelligent subsampling of ANI-1x |

**Why Essential:**
- Near-exact coupled-cluster accuracy
- Enables transfer learning from DFT to CCSD(T)
- Unique high-accuracy benchmark
- Critical for chemical accuracy validation

**Reference:**
> Smith, J.S. et al. "Approaching coupled cluster accuracy with a general-purpose neural network potential through transfer learning." Nature Communications 10, 2903 (2019).

**Implementation Notes:**
- 10% of ANI-1x intelligently selected
- Same molecular identifiers as ANI-1x
- Ideal for Δ-learning approaches

---

### 3. GEOM (Geometric Ensemble Of Molecules)

| Property | Value |
|----------|-------|
| **Priority** | 🔴 HIGHEST |
| **Size** | 37 million conformations |
| **Molecules** | 450,000+ unique molecules |
| **Level of Theory** | GFN2-xTB + DFT single points |
| **Properties** | Energies, statistical weights, dipoles, charges |
| **Source** | QM9 + AICures drug-like molecules |

**Why Essential:**
- Largest conformer ensemble dataset available
- Essential for conformer generation models
- Bridges QM9 (small) and drug-like (large) molecules
- DFT-quality statistical weights for 1,511 BACE species

**Reference:**
> Axelrod, S. & Gómez-Bombarelli, R. "GEOM, energy-annotated molecular conformations for property prediction and molecular generation." Scientific Data 9, 185 (2022).

**Implementation Notes:**
- Two subsets: `qm9` and `drugs`
- MessagePack format (convert to NPZ)
- Average 13 conformers/molecule (QM9), 100+ (drugs)

---

### 4. SPICE (Small-molecule/Protein Interaction Chemical Energies)

| Property | Value |
|----------|-------|
| **Priority** | 🔴 HIGHEST |
| **Size** | 1.1+ million conformations |
| **Molecules** | Diverse small molecules, dimers, dipeptides |
| **Level of Theory** | ωB97M-D3(BJ)/def2-TZVPPD |
| **Properties** | Energies, forces, multipoles, bond orders |
| **Elements** | 17 (H, Li, B, C, N, O, F, Na, Mg, Si, P, S, Cl, K, Ca, Br, I) |

**Why Essential:**
- Drug-protein interaction focus
- High-quality DFT with dispersion correction
- Forces AND energies (critical for MD)
- Includes charged molecules and solvation
- Covers covalent AND non-covalent interactions

**Reference:**
> Eastman, P. et al. "SPICE, A Dataset of Drug-like Molecules and Peptides for Training Machine Learning Potentials." Scientific Data 10, 11 (2023).

**Implementation Notes:**
- HDF5 format available
- Subsets: PubChem, dipeptides, amino acids, ion pairs, water clusters
- Version 2.0 doubles the data

---

### 5. OC20 (Open Catalyst 2020)

| Property | Value |
|----------|-------|
| **Priority** | 🔴 HIGHEST |
| **Size** | 1,281,040 DFT relaxations (~265M single points) |
| **Systems** | 82 adsorbates on 11,451 catalyst surfaces |
| **Level of Theory** | RPBE DFT |
| **Properties** | Energies, forces, relaxation trajectories |
| **Elements** | 55 unique elements |

**Why Essential:**
- Largest catalyst dataset ever created
- Defines GNN benchmarks for materials science
- Meta AI / CMU collaboration (industry backing)
- Essential for renewable energy applications
- Includes train/val/test splits with OOD evaluation

**Reference:**
> Chanussot, L. et al. "Open Catalyst 2020 (OC20) Dataset and Community Challenges." ACS Catalysis 11, 6059-6072 (2021).

**Implementation Notes:**
- LMDB format (convert to NPZ)
- Periodic boundary conditions
- Tasks: S2EF, IS2RE, IS2RS

---

## Tier 2: HIGH PRIORITY (Competitive Differentiation)

These datasets provide **significant competitive advantage** and should be implemented after Tier 1.

---

### 6. MD17 / rMD17 (Revised MD17)

| Property | Value |
|----------|-------|
| **Priority** | 🟠 HIGH |
| **Size** | ~100,000 structures per molecule |
| **Molecules** | 10 small organic molecules |
| **Level of Theory** | PBE/def2-SVP (rMD17), CCSD(T) available |
| **Properties** | Energies, forces |
| **Source** | Ab initio molecular dynamics at 500K |

**Why High Priority:**
- Standard benchmark for force field validation
- Forces included (critical for MD)
- CCSD(T) versions available for 5 molecules
- Direct comparison with all published NNP methods

**Molecules:** Benzene, Uracil, Naphthalene, Aspirin, Salicylic acid, Malonaldehyde, Ethanol, Toluene, Paracetamol, Azobenzene

**Reference:**
> Christensen, A.S. & von Lilienfeld, O.A. "On the role of gradients for machine learning of molecular energies and forces." Machine Learning: Science and Technology 1, 045018 (2020).

**Implementation Notes:**
- Use rMD17 (revised) - original has numerical noise
- NPZ format available
- **WARNING:** Do not train on >1000 samples (autocorrelation)

---

### 7. QMugs (Quantum Mechanical Properties of Drug-like Molecules)

| Property | Value |
|----------|-------|
| **Priority** | 🟠 HIGH |
| **Size** | 665,000+ molecules, ~2M conformers |
| **Source** | ChEMBL database (bioactive compounds) |
| **Level of Theory** | GFN2-xTB + ωB97X-D/def2-SVP DFT |
| **Properties** | Energies, charges, dipoles, orbital matrices |
| **Data Size** | 7+ TB uncompressed (includes wavefunctions) |

**Why High Priority:**
- Drug-like molecules from ChEMBL
- Biological activity annotations included
- DFT wavefunctions (density/orbital matrices)
- Bridges QM calculations and drug discovery
- Significantly larger molecules than QM9

**Reference:**
> Isert, C. et al. "QMugs, quantum mechanical properties of drug-like molecules." Scientific Data 9, 273 (2022).

**Implementation Notes:**
- SDF format with properties
- 3 conformers per molecule
- Keep same ChEMBL-ID conformers in same split

---

### 8. ANI-2x

| Property | Value |
|----------|-------|
| **Priority** | 🟠 HIGH |
| **Size** | Extended from ANI-1x |
| **Elements** | H, C, N, O, F, Cl, S (7 elements) |
| **Level of Theory** | ωB97X/6-31G(d) DFT |
| **Properties** | Energies, forces |

**Why High Priority:**
- Covers ~90% of drug-like molecules
- Adds halogens (F, Cl) and sulfur
- Widely used in pharmaceutical industry
- Direct extension of ANI-1x methodology

**Reference:**
> Devereux, C. et al. "Extending the Applicability of the ANI Deep Learning Molecular Potential to Sulfur and Halogens." J. Chem. Theory Comput. 16, 4192-4202 (2020).

**Implementation Notes:**
- Same format as ANI-1x
- Includes torsion and dimer sampling

---

### 9. OC22 (Open Catalyst 2022)

| Property | Value |
|----------|-------|
| **Priority** | 🟠 HIGH |
| **Size** | 62,331 DFT relaxations (~9.85M single points) |
| **Systems** | 4,728 oxide surfaces with OER intermediates |
| **Level of Theory** | PBE+U with spin polarization |
| **Properties** | Energies, forces, magnetic moments |

**Why High Priority:**
- Oxide electrocatalysts (complements OC20 metals)
- Oxygen evolution reaction (OER) focus
- Spin-polarized calculations
- Critical for water splitting / green hydrogen

**Reference:**
> Tran, R. et al. "The Open Catalyst 2022 (OC22) Dataset and Challenges for Oxide Electrocatalysts." ACS Catalysis 13, 3066-3084 (2023).

**Implementation Notes:**
- Same infrastructure as OC20
- Includes magnetic moment data

---

### 10. Tox21

| Property | Value |
|----------|-------|
| **Priority** | 🟠 HIGH |
| **Size** | ~8,000 compounds |
| **Tasks** | 12 toxicity endpoints (binary classification) |
| **Endpoints** | 7 nuclear receptors, 5 stress responses |
| **Source** | NIH Tox21 Data Challenge |

**Why High Priority:**
- Industry-standard toxicity benchmark
- MoleculeNet cornerstone dataset
- Essential for drug safety prediction
- Regulatory relevance (FDA)

**Endpoints:** NR-AR, NR-AhR, NR-AR-LBD, NR-ER, NR-ER-LBD, NR-Aromatase, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53

**Reference:**
> Tox21 Data Challenge (2014), integrated into MoleculeNet

**Implementation Notes:**
- SMILES-based (no 3D coordinates in original)
- Missing labels common (use masking)
- Scaffold splitting recommended

---

## Tier 3: STRATEGIC (Expands Market Reach)

These datasets **expand MILIA's applicability** to additional domains and use cases.

---

### 11. GEOM-Drugs

| Property | Value |
|----------|-------|
| **Priority** | 🟡 MEDIUM-HIGH |
| **Size** | 317,000+ molecules, 430k with conformers |
| **Molecules** | Drug-like from AICures |
| **Avg Size** | 44 atoms (up to 181 atoms) |

**Why Strategic:** Larger molecules than QM9; drug discovery focus; 3D generative model benchmark

---

### 12. ToxCast

| Property | Value |
|----------|-------|
| **Priority** | 🟡 MEDIUM-HIGH |
| **Size** | 8,615 compounds |
| **Tasks** | 600+ in vitro assays |

**Why Strategic:** Extends Tox21; high-throughput screening; regulatory applications

---

### 13. ESOL (Aqueous Solubility)

| Property | Value |
|----------|-------|
| **Priority** | 🟡 MEDIUM-HIGH |
| **Size** | 1,128 compounds |
| **Task** | Regression (log solubility) |

**Why Strategic:** Key pharmaceutical property; solubility prediction standard

---

### 14. FreeSolv (Solvation Free Energy)

| Property | Value |
|----------|-------|
| **Priority** | 🟡 MEDIUM-HIGH |
| **Size** | 643 compounds |
| **Task** | Regression (hydration free energy) |

**Why Strategic:** Experimental thermodynamic data; force field validation

---

### 15. Lipophilicity

| Property | Value |
|----------|-------|
| **Priority** | 🟡 MEDIUM-HIGH |
| **Size** | 4,200 compounds |
| **Task** | Regression (octanol-water partition) |

**Why Strategic:** Key ADMET property; drug absorption prediction

---

### 16. xxMD (Extended Excited-state MD)

| Property | Value |
|----------|-------|
| **Priority** | 🟡 MEDIUM |
| **Size** | 4 molecules with reactive trajectories |
| **Molecules** | Azobenzene, malonaldehyde, stilbene, dithiophene |
| **Level of Theory** | SA-CASSCF + M06 DFT |

**Why Strategic:** Reactive/nonadiabatic dynamics; extends MD17; tests extrapolation to reactions

**Reference:**
> "Beyond MD17: the reactive xxMD dataset." Scientific Data (2024).

---

### 17. BACE (β-Secretase Inhibitors)

| Property | Value |
|----------|-------|
| **Priority** | 🟡 MEDIUM |
| **Size** | 1,522 compounds |
| **Task** | Binary classification |

**Why Strategic:** Alzheimer's drug target; scaffold splitting benchmark

---

### 18. BBBP (Blood-Brain Barrier Penetration)

| Property | Value |
|----------|-------|
| **Priority** | 🟡 MEDIUM |
| **Size** | 2,000+ compounds |
| **Task** | Binary classification |

**Why Strategic:** CNS drug development; membrane permeability

---

### 19. HIV

| Property | Value |
|----------|-------|
| **Priority** | 🟡 MEDIUM |
| **Size** | 41,127 compounds |
| **Task** | Binary classification |

**Why Strategic:** Large classification dataset; antiviral drug discovery

---

### 20. ClinTox

| Property | Value |
|----------|-------|
| **Priority** | 🟡 MEDIUM |
| **Size** | 1,491 drugs |
| **Tasks** | FDA approval + clinical trial toxicity |

**Why Strategic:** Clinical trial outcomes; regulatory decision support

---

## Tier 4: SPECIALIZED (Domain Expansion)

These datasets serve **specific applications** and specialized user communities.

---

### 21. QDπ

| Property | Value |
|----------|-------|
| **Priority** | 🟢 MEDIUM |
| **Size** | 1.6 million structures |
| **Level of Theory** | ωB97M-D3(BJ)/def2-TZVPPD |

**Why Specialized:** Active learning curated; drug fragments; relative conformational energies

**Reference:**
> "The QDπ dataset, training data for drug-like molecules and biopolymer fragments." Scientific Data (2025).

---

### 22. SIDER (Side Effect Resource)

| Property | Value |
|----------|-------|
| **Priority** | 🟢 MEDIUM |
| **Size** | 1,427 drugs |
| **Tasks** | 27 side effect categories |

**Why Specialized:** Adverse drug reactions; pharmacovigilance

---

### 23. PDBbind

| Property | Value |
|----------|-------|
| **Priority** | 🟢 MEDIUM |
| **Size** | ~20,000 protein-ligand complexes |
| **Task** | Binding affinity regression |

**Why Specialized:** Structure-based drug design; docking validation

---

### 24. MUV (Maximum Unbiased Validation)

| Property | Value |
|----------|-------|
| **Priority** | 🟢 LOW-MEDIUM |
| **Size** | 93,127 compounds |
| **Tasks** | 17 targets |

**Why Specialized:** Virtual screening benchmark; designed to avoid bias

---

### 25. PCBA (PubChem BioAssay)

| Property | Value |
|----------|-------|
| **Priority** | 🟢 LOW-MEDIUM |
| **Size** | 437,929 compounds |
| **Tasks** | 128 bioassays |

**Why Specialized:** Large multi-task; high-throughput screening

---

### 26. OMat24 (Open Materials 2024)

| Property | Value |
|----------|-------|
| **Priority** | 🟢 MEDIUM (EMERGING) |
| **Size** | 1.6M+ examples |
| **Source** | Materials Project relaxations |

**Why Specialized:** Inorganic materials; foundation model training; Meta AI release

---

## Implementation Roadmap

### Phase 1: Foundation (Months 1-3)
1. ✅ QM9 (COMPLETE)
2. ANI-1x
3. ANI-1ccx
4. MD17/rMD17

### Phase 2: Drug Discovery (Months 4-6)
5. SPICE
6. QMugs
7. ANI-2x
8. Tox21

### Phase 3: Materials & Conformers (Months 7-9)
9. GEOM
10. OC20
11. OC22

### Phase 4: ADMET Suite (Months 10-12)
12. ESOL
13. FreeSolv
14. Lipophilicity
15. BBBP
16. BACE

### Phase 5: Extended Coverage (Year 2)
17-26. Remaining Tier 3 & 4 datasets

---

## Technical Considerations

### Common Implementation Patterns

Based on QM9 implementation experience, each new dataset requires:

1. **Dataset Class** (`datasets/implementations/`)
   - `DatasetMetadata`: name, version, description
   - `DatasetSchema`: required/optional properties, **identifier_keys (InChI first!)**
   - `DatasetFeatures`: feature support flags
   - Abstract method implementations

2. **Handler Class** (`handlers/dataset_handlers.py`)
   - Only if custom validation/processing needed
   - Implements `get_identifier_keys()` method

3. **Configuration** (`config.yaml`)
   - Dataset-specific config section
   - Preprocessing settings if needed

4. **Preprocessor** (if source not NPZ)
   - Convert source format → NPZ
   - Standardize property names

### Key Lessons from QM9

| Issue | Solution |
|-------|----------|
| Identifier order matters | **Always InChI first** in `identifier_keys` |
| Handler vs Dataset class | Schema defined in Dataset class, not handler |
| Registry registration | `@register` decorator + `__init__.py` import |

---

## Competitive Analysis

### Datasets Supported by Competitors

| Dataset | DeepChem | PyG | SchNetPack | **MILIA (Target)** |
|---------|----------|-----|------------|-------------------|
| QM9 | ✅ | ✅ | ✅ | ✅ |
| ANI-1x | ❌ | ❌ | ✅ | 🎯 |
| GEOM | ❌ | ✅ | ❌ | 🎯 |
| SPICE | ❌ | ❌ | ❌ | 🎯 |
| OC20 | ❌ | ✅ | ❌ | 🎯 |
| MD17 | ✅ | ✅ | ✅ | 🎯 |
| Tox21 | ✅ | ❌ | ❌ | 🎯 |
| QMugs | ❌ | ❌ | ❌ | 🎯 |

**MILIA Differentiation:** By implementing all Tier 1 and Tier 2 datasets, MILIA would become the **most comprehensive** molecular ML platform, covering domains no single competitor currently addresses.

---

## Document Information

**Version:** 1.0.0  
**Created:** 2025-12-27  
**Author:** MILIA Development Team  
**Evidence Sources:** Web searches of Scientific Data, Nature Communications, ACS Catalysis, arXiv, GitHub repositories, and benchmark documentation

**Key References:**
- MoleculeNet: Wu, Z. et al. Chem. Sci. 9, 513-530 (2018)
- ANI: Smith, J.S. et al. Chem. Sci. 8, 3192-3203 (2017)
- OC20: Chanussot, L. et al. ACS Catalysis 11, 6059-6072 (2021)
- GEOM: Axelrod, S. et al. Scientific Data 9, 185 (2022)
- SPICE: Eastman, P. et al. Scientific Data 10, 11 (2023)
- QMugs: Isert, C. et al. Scientific Data 9, 273 (2022)
