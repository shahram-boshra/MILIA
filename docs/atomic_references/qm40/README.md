# QM40 — atomic reference inputs (Gaussian 16)

**Status:** inputs ready, not yet computed. Until the outputs are added, QM40 does not compute
`Internal_E_0K_ATOM`.

**Dataset level of theory** (Madushanka, Moura, Kraka, *Sci. Data* 11 (2024), doi:10.1038/s41597-024-04206-y):
B3LYP/6-31G(2df,p) in Gaussian 16 for optimizations and frequencies; elements H, C, N, O, F, S, Cl. The
paper states no other keywords, so these inputs use Gaussian 16's defaults, written out explicitly. The
paper publishes no atomic references.

| Setting | Value | Source |
|---|---|---|
| Method / basis | B3LYP/6-31G(2df,p), unrestricted (`UB3LYP`) for open-shell atoms | paper; unrestricted as for QM9's atomrefs |
| Grid | `Int=UltraFine` | Gaussian 16 default |
| SCF | `SCF=Tight` (1e-8 on the density) | Gaussian 16 default |
| Lowest SCF solution | `Stable=Opt` | MILIA (atoms are open-shell) |
| Thermochemistry | `Freq` in a second step (`Geom=Check Guess=Read`) | as the molecules' U0/U/H/G |
| Multiplicity | H 2, C 3, N 4, O 3, F 2, S 3, Cl 2 | atomic ground states |

**Run** (Gaussian 16, from this directory; record the exact revision, e.g. C.01, from the log header):

```bash
mkdir -p outputs
for f in inputs/*.gjf; do el=$(basename "$f" .gjf); g16 < "$f" > "outputs/$el.log"; done
python ../parse_outputs.py . --program gaussian
```

**Then** set `qm40_config.atomic_references` in `configs/datasets/qm40.yaml` from `results.json`, rounded to
1e-10 Eh: `source: milia-computed`; `program: Gaussian 16 Rev. <revision>`; energy tables
`Internal_E_0K` (= `U0`), `Inter_E_298` (= `U298`), `Enthalpy` (= `H298`), `Free_E` (= `G298`). Commit
`outputs/` (raw logs, verbatim) and `results.json`.
