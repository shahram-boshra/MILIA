"""Isolated-atom reference energies at the xxMD-DFT level of theory (UKS M06/6-31G, Psi4).

xxMD-DFT energies are spin-polarized (unrestricted) Kohn-Sham M06/6-31G single points computed with
Psi4 (Pengmei, Liu, Shu, Sci. Data 11, 222 (2024), SI section G.3); the dataset publishes no atomic
references. This script computes each element's isolated atom at its ground-state multiplicity with
Psi4's default DFT grid, from three initial guesses, and keeps the lowest energy.

Run in the environment locked by ``conda-linux-64.lock`` (Psi4 1.11, conda-forge):

    micromamba create -p ./psi4env --file conda-linux-64.lock
    ./psi4env/bin/python compute_atomic_references.py

Writes ``outputs/<element>_<guess>.out`` (Psi4 output) and ``results.json``; the selected energies
are ``configs/datasets/xxmd.yaml`` ``xxmd_config.atomic_references.energies_hartree.energy``.
"""

from __future__ import annotations

import json
from pathlib import Path

import psi4

HERE = Path(__file__).resolve().parent

# Element -> (atomic number, ground-state spin multiplicity 2S+1).
ATOMS = {"H": (1, 2), "C": (6, 3), "N": (7, 4), "O": (8, 3), "S": (16, 3)}
GUESSES = ("sad", "core", "gwh")
OPTIONS = {
    "basis": "6-31g",
    "reference": "uks",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
    "maxiter": 300,
}
METHOD = "m06"


def main() -> None:
    outputs = HERE / "outputs"
    outputs.mkdir(exist_ok=True)
    psi4.set_memory("2 GB")
    results = {
        "psi4_version": psi4.__version__,
        "method": METHOD,
        "options": OPTIONS,
        "atoms": {},
    }
    for symbol, (z, multiplicity) in ATOMS.items():
        energies = {}
        for guess in GUESSES:
            psi4.core.clean()
            psi4.core.clean_options()
            psi4.core.set_output_file(str(outputs / f"{symbol}_{guess}.out"), False)
            psi4.geometry(f"0 {multiplicity}\n{symbol} 0 0 0\nsymmetry c1\nnoreorient\nnocom")
            psi4.set_options({**OPTIONS, "guess": guess})
            energies[guess] = psi4.energy(METHOD)
        selected = min(energies, key=energies.get)
        results["atoms"][symbol] = {
            "Z": z,
            "multiplicity": multiplicity,
            "energies_hartree": energies,
            "selected_guess": selected,
            "selected_energy_hartree": energies[selected],
            "guess_spread_hartree": max(energies.values()) - min(energies.values()),
        }
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
