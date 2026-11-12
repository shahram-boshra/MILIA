# ANI-2x — atomic reference inputs (Gaussian 09)

**Status:** inputs ready, not yet computed; **the training data's program version is not published**
(below). Until it is confirmed, ANI-2x does not compute `energy_ATOM`.

**Dataset level of theory**: ωB97X/6-31G\* (Devereux et al., *J. Chem. Theory Comput.* 16, 4192 (2020),
doi:10.1021/acs.jctc.0c00121; data release Huddleston et al., doi:10.5281/zenodo.10108942, `wB97X/631Gd`);
elements H, C, N, O, S, F, Cl. The paper's "using Gaussian 09 and Gaussian 16" (Methods, "Building a
Database of S-, F-, and Cl-Containing Molecules") describes the reference calculations of its *test sets*,
and its torsion scans name Gaussian 09 Rev. C.01/D.01 and Gaussian 16; neither states the program of the
training data. The modelforge curation of the release (doi:10.5281/zenodo.11642744, not by the dataset's
authors) states Gaussian 09 without a revision. Gaussian 09 and 16 use different default grids (FineGrid /
UltraFine), so these Gaussian 09 inputs follow the only available statement and are run only once the
authors confirm the program and revision. No atomic references are published.

| Setting | Value | Source |
|---|---|---|
| Method / basis | ωB97X/6-31G\*, unrestricted (`UwB97X`) | paper |
| Grid | `Int=FineGrid` | Gaussian 09 default (use `Int=UltraFine` if Gaussian 16 is confirmed) |
| SCF | `SCF=Tight` (1e-8 on the density) | Gaussian 09/16 default |
| Lowest SCF solution | `Stable=Opt` | MILIA (atoms are open-shell) |
| Multiplicity | H 2, C 3, N 4, O 3, F 2, S 3, Cl 2 | atomic ground states |

**Run** (Gaussian 09, from this directory):

```bash
mkdir -p outputs
for f in inputs/*.gjf; do el=$(basename "$f" .gjf); g09 < "$f" > "outputs/$el.log"; done
python ../parse_outputs.py . --program gaussian
```

**Then** set `ani2x_config.atomic_references` (`source: milia-computed`, program with revision, energy
table `energy` = `scf`, rounded to 1e-10 Eh). Commit `outputs/` (verbatim) and `results.json`.
