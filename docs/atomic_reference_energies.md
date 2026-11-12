# Atomic reference energies

An **atomization energy** (`energy_ATOM`, `U0_ATOM`, …) is a molecule's total energy minus the energies
of its isolated atoms. Both must come from the **same level of theory**: method, basis set, spin
treatment and the program's implementation of the method. Functionals of the same name can differ
between programs (B3LYP uses VWN(RPA) correlation in Gaussian and VWN5 in many other codes;
Lehtola and Marques, *J. Chem. Phys.* 159 (2023), doi:10.1063/5.0167763). Mixing levels gives
a quantity that is not an atomization energy.

## Policy

MILIA computes an atomization energy only from references declared for that dataset, in this order:

1. **author-published**: the dataset's authors published isolated-atom energies at the dataset's
   level of theory (MILIA copies them verbatim and cites them);
2. **milia-computed**: none are published, and MILIA can reproduce the exact level of theory in the
   same program; MILIA computes them, and ships the inputs, a locked environment and the raw outputs;
3. **user-supplied**: set by the user in `<dataset>_config.atomic_references` (optional; never
   required).

If none applies (for example, the original program is not available), the atomization energy is
**not computed**, and the reason is logged and saved. Total energies and forces never depend on these
references.

Each dataset declares its references in `configs/datasets/<dataset>.yaml`:

```yaml
qm9_config:
  atomic_references:
    source: author-published      # author-published | milia-computed | user-supplied
    level_of_theory: B3LYP/6-31G(2df,p), spin-unrestricted atoms
    program: Gaussian 09
    citation: Ramakrishnan et al., Sci. Data 1, 140022 (2014); atomref.txt
    energies_hartree:             # one table per energy property, atomic number -> Hartree
      U0: {1: -0.500273, 6: -37.846772, 7: -54.583861, 8: -75.064579, 9: -99.718730}
```

A table applies only to the energy property it is listed under (`calculate_atomization_energy_from`).
Processing saves `<processed file stem>.atomic_references.json` next to the processed file, recording
the source, level of theory, program, citation and the energies used, or why the atomization energy
was not computed. In Python: `milia_pipeline.config.get_dataset_atomic_references(dataset_type)`.

The global `atomic_energies_hartree` in `configs/main.yaml` is deprecated (no recorded level of
theory or source), unused by MILIA, and removed in 2.0.0.

## Datasets

| Dataset | Level of theory | References | Source |
|---|---|---|---|
| QM9 | B3LYP/6-31G(2df,p), Gaussian 09 | U0, U, H, G for H, C, N, O, F | author-published: `atomref.txt` (figshare file 3195395) |
| xxMD-DFT | UKS M06/6-31G, Psi4 | `energy` for H, C, N, O, S | milia-computed (below) |
| DFT (VQM24) | ωB97X-D3/cc-pVDZ, Psi4 1.7 | none needed | author-published atomization energy per molecule (`Eatomization`); MILIA computes none |
| QM40 | B3LYP/6-31G(2df,p), Gaussian 16 | — | not computed; inputs ready: `docs/atomic_references/qm40/` |
| ANI-1x | ωB97X/6-31G\*, Gaussian 09 | — | not computed; inputs ready: `docs/atomic_references/ani1x/` |
| ANI-2x | ωB97X/6-31G\*; training-data program not published (third-party curation: Gaussian 09) | — | not computed; Gaussian 09 inputs ready, run once the authors confirm the program: `docs/atomic_references/ani2x/` |
| rMD17 | PBE/def2-SVP, ORCA 4.0.1 | — | not computed; inputs ready: `docs/atomic_references/rmd17/` |
| ANI-1ccx | CCSD(T)\*/CBS composite, ORCA | — | not computed: ORCA version, auxiliary bases, frozen core and SCF settings unpublished (SI S1.1): `docs/atomic_references/ani1ccx/` |
| QDπ | ωB97M-D3(BJ)/def2-TZVPPD, Psi4 1.7 | `energy` for H, Li, C, N, O, F, Na, P, S, Cl, K, Br, I | milia-computed in Psi4 1.7; neutral molecules only: `docs/atomic_references/qdpi/` |

None of these datasets publishes isolated-atom energies. Where MILIA cannot run the original program,
`docs/atomic_references/<dataset>/` holds the exact inputs (each setting traced to the paper or to the
program's default), the run commands and `parse_outputs.py`, which writes `results.json` for the
dataset's `atomic_references`.

### xxMD-DFT (milia-computed)

xxMD-DFT energies are spin-polarized (unrestricted) Kohn-Sham M06/6-31G single points from Psi4
(Pengmei, Liu, Shu, *Sci. Data* 11, 222 (2024), SI section G.3); the dataset publishes no atomic
references and names no Psi4 version.

Method, as run by `docs/atomic_references/xxmd/compute_atomic_references.py`:

- Psi4 1.11 from conda-forge, environment locked in `conda-linux-64.lock` (libxc 7.1.2);
- `reference uks`, M06, `basis 6-31g`; otherwise Psi4 defaults: density-fitted SCF (`scf_type df`,
  cc-pVDZ-JKFIT auxiliary basis), Treutler radial and nuclear schemes, 75 radial × 302 spherical
  points, no pruning;
- one atom at the origin, `symmetry c1`, ground-state multiplicity: H 2, C 3, N 4, O 3, S 3;
- `e_convergence 1e-10`, `d_convergence 1e-8`; initial guesses SAD, CORE and GWH; the lowest energy
  is kept (largest spread between guesses: 2.2e-7 Eh, carbon); ⟨S²⟩ within 0.0042 of S(S+1).

Values in `configs/datasets/xxmd.yaml` are `results.json` rounded to 1e-10 Eh. Rerunning reproduced
them to 2e-9 Eh (thread count). The raw Psi4 outputs are in `outputs/`, verbatim.

To reproduce:

```bash
cd docs/atomic_references/xxmd
micromamba create -p ./psi4env --file conda-linux-64.lock
./psi4env/bin/python compute_atomic_references.py
```
