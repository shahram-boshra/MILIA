"""Isolated-atom reference energies at the QDπ level of theory (ωB97M-D3(BJ)/def2-TZVPPD, Psi4 1.7).

QDπ energies are ωB97M-D3(BJ)/def2-TZVPPD "as implemented in the PSI4 v1.7 software" (Zeng, Giese,
Götz, York, Sci. Data 12, 693 (2025)); the dataset publishes no atomic references and no other
settings, so Psi4 1.7's defaults apply. This script computes each element's neutral isolated atom
(unrestricted Kohn-Sham, ground-state multiplicity) from three initial guesses and keeps the lowest
energy. The D3(BJ) correction of a single atom is zero.

Run in the environment locked by ``conda-linux-64.lock`` (Psi4 1.7 from the psi4 channel):

    micromamba create -p ./psi4env --file conda-linux-64.lock
    ./psi4env/bin/python compute_atomic_references.py

Writes ``outputs/<element>_<guess>.out`` (Psi4 output) and ``results.json``; the selected energies
are ``configs/datasets/qdpi.yaml`` ``qdpi_config.atomic_references.energies_hartree.energy``.
"""

from __future__ import annotations

import json
from pathlib import Path

import psi4

HERE = Path(__file__).resolve().parent

# Element -> (atomic number, ground-state spin multiplicity 2S+1).
ATOMS = {
    "H": (1, 2),
    "Li": (3, 2),
    "C": (6, 3),
    "N": (7, 4),
    "O": (8, 3),
    "F": (9, 2),
    "Na": (11, 2),
    "P": (15, 4),
    "S": (16, 3),
    "Cl": (17, 2),
    "K": (19, 2),
    "Br": (35, 2),
    "I": (53, 2),
}
GUESSES = ("sad", "core", "gwh")
OPTIONS = {
    "basis": "def2-tzvppd",
    "reference": "uks",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
    "maxiter": 300,
}
METHOD = "wb97m-d3bj"


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
