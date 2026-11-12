# QDπ — atomic references (Psi4 1.7, computed)

**Status:** computed; declared in `configs/datasets/qdpi.yaml` (`qdpi_config.atomic_references`,
`source: milia-computed`). QDπ computes `energy_ATOM` for **neutral** molecules only: an atomization
energy dissociates a neutral molecule into neutral ground-state atoms, and a charged molecule's products
would include an ion.

**Dataset level of theory** (Zeng, Giese, Götz, York, *Sci. Data* 12, 693 (2025),
doi:10.1038/s41597-025-04972-3; data doi:10.5281/zenodo.14970869): ωB97M-D3(BJ)/def2-TZVPPD "as
implemented in the PSI4 v1.7 software"; 13 elements (H, Li, C, N, O, F, Na, P, S, Cl, K, Br, I); neutral
and charged subsets; energies in eV (MILIA's preprocessing converts them to Hartree). The paper states no
other settings and publishes no atomic references.

| Setting | Value | Source |
|---|---|---|
| Program | Psi4 1.7 (`1.7+6ce35a5`, psi4 channel), locked in `conda-linux-64.lock` | paper (version) |
| Method / basis | `wb97m-d3bj` / `def2-tzvppd`, `reference uks` | paper; unrestricted for open-shell atoms |
| SCF / grid | density fitting (`scf_type df`), 75 radial × 302 spherical points | Psi4 1.7 defaults (paper states none) |
| Iodine | def2 ECP (28 core electrons), as the basis defines it | def2-TZVPPD |
| Convergence | `e_convergence 1e-10`, `d_convergence 1e-8` | MILIA (tighter than the defaults) |
| Lowest SCF solution | SAD, CORE and GWH initial guesses; lowest kept | MILIA (as for xxMD) |
| Multiplicity | H, Li, F, Na, Cl, K, Br, I 2; C, O, S 3; N, P 4 | atomic ground states |

**Results** (`results.json`, raw outputs in `outputs/`, verbatim): ⟨S²⟩ within 0.0045 of S(S+1) for every
atom; guesses agree within 2.2e-7 Eh except Li, whose CORE guess converged to the excited ²P state
(+0.071 Eh, the 2s→2p excitation) and is not selected. The D3(BJ) correction of a single atom is zero.
Psi4 1.7 prints that its ECP support is in beta; the dataset was computed with the same version.

**Reproduce** (from this directory):

```bash
micromamba create -p ./psi4env --file conda-linux-64.lock
PATH=./psi4env/bin:$PATH ./psi4env/bin/python compute_atomic_references.py
```
