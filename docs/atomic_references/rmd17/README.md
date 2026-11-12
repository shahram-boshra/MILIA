# rMD17 — atomic reference inputs (ORCA 4.0.1)

**Status:** inputs ready, not yet computed. Until the outputs are added, rMD17 does not compute
`energies_ATOM`.

**Dataset level of theory** (Christensen, von Lilienfeld, *Mach. Learn.: Sci. Technol.* (2020),
doi:10.1088/2632-2153/abba6f; arXiv:2007.09593): "All calculations were performed in ORCA 4.0.1 using the
PBE functional and the def2-SVP basis set with the resolution-of-identity (RI) approximation for the Coulomb
integrals", "VeryTightSCF", "Grid7 and NoFinalGrid"; molecules contain H, C, N, O. No atomic references
are published.

| Setting | Value | Source |
|---|---|---|
| Method / basis | PBE/def2-SVP, unrestricted (`UKS`) | paper |
| RI-J | `RI def2/J` | paper (RI for Coulomb); def2/J is ORCA's auxiliary basis for def2 RI-J |
| SCF / grid | `VeryTightSCF Grid7 NoFinalGrid` | paper |
| Lowest SCF solution | three initial guesses (`PModel`, `HCore`, `Hueckel`); lowest kept | MILIA (as for xxMD) |
| Multiplicity | H 2, C 3, N 4, O 3 | atomic ground states |

**Run** (ORCA 4.0.1, free academic licence; from this directory):

```bash
mkdir -p outputs
for f in inputs/*.inp; do orca "$f" > "outputs/$(basename "$f" .inp).out"; done
python ../parse_outputs.py . --program orca
```

**Then** set `rmd17_config.atomic_references` (`source: milia-computed`, `program: ORCA 4.0.1`, energy table
`energies` = `selected_energy_hartree`, rounded to 1e-10 Eh). Commit `outputs/` (verbatim) and
`results.json`.
