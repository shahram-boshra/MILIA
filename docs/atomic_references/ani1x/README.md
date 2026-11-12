# ANI-1x — atomic reference inputs (Gaussian 09)

**Status:** inputs ready, not yet computed. Until the outputs are added, ANI-1x does not compute
`energy_ATOM`.

**Dataset level of theory** (Smith et al., *Sci. Data* 7 (2020), doi:10.1038/s41597-020-0473-z):
"All wB97x/6-31G\* calculations were carried out in the Gaussian 09 electronic structure package"
(Revision E.01 cited); elements H, C, N, O. No other keywords and no atomic references are published
(TorchANI's ground-state atomic energies document no program or settings, so they are not used).

| Setting | Value | Source |
|---|---|---|
| Method / basis | ωB97X/6-31G\*, unrestricted (`UwB97X`) | paper |
| Grid | `Int=FineGrid` | Gaussian 09 default |
| SCF | `SCF=Tight` (1e-8 on the density) | Gaussian 09 default |
| Lowest SCF solution | `Stable=Opt` | MILIA (atoms are open-shell) |
| Multiplicity | H 2, C 3, N 4, O 3 | atomic ground states |

**Run** (Gaussian 09 Rev. E.01, from this directory):

```bash
mkdir -p outputs
for f in inputs/*.gjf; do el=$(basename "$f" .gjf); g09 < "$f" > "outputs/$el.log"; done
python ../parse_outputs.py . --program gaussian
```

**Then** set `ani1x_config.atomic_references` in `configs/datasets/ani1x.yaml` (`source: milia-computed`,
`program: Gaussian 09 Rev. E.01`, energy table `energy` = `scf`, rounded to 1e-10 Eh). Commit `outputs/`
(verbatim) and `results.json`.
