"""Collect isolated-atom energies from Gaussian or ORCA outputs into ``results.json``.

Usage (from the repository root, after running a dataset's inputs as described in its README):

    python docs/atomic_references/parse_outputs.py docs/atomic_references/<dataset> \\
        --program gaussian|orca

Reads ``<dataset>/outputs/*.log`` (Gaussian) or ``*.out`` (ORCA) and writes ``<dataset>/results.json``
with, per element, every energy found and the selected one. Gaussian: the energy of the last
``SCF Done`` (after ``Stable=Opt``) and, when the job ran ``Freq``, the thermochemical sums (keys
``U0``, ``U298``, ``H298``, ``G298``). ORCA: ``FINAL SINGLE POINT ENERGY`` of each initial guess; the
lowest is selected. Every job must have terminated normally. The values then go, rounded to 1e-10
Eh, into ``<dataset>_config.atomic_references`` (``docs/atomic_reference_energies.md``).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

GAUSSIAN_SCF = re.compile(r"SCF Done:\s+E\(\S+\)\s+=\s+(-?\d+\.\d+)")
GAUSSIAN_THERMO = {
    "U0": re.compile(r"Sum of electronic and zero-point Energies=\s+(-?\d+\.\d+)"),
    "U298": re.compile(r"Sum of electronic and thermal Energies=\s+(-?\d+\.\d+)"),
    "H298": re.compile(r"Sum of electronic and thermal Enthalpies=\s+(-?\d+\.\d+)"),
    "G298": re.compile(r"Sum of electronic and thermal Free Energies=\s+(-?\d+\.\d+)"),
}
GAUSSIAN_OK = "Normal termination of Gaussian"
ORCA_ENERGY = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
ORCA_OK = "****ORCA TERMINATED NORMALLY****"


def _gaussian(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    links = text.count("Initial command:")
    if text.count(GAUSSIAN_OK) < max(links, 1):
        raise SystemExit(f"{path}: not every Gaussian job terminated normally")
    energies = {"scf": float(GAUSSIAN_SCF.findall(text)[-1])}
    for key, pattern in GAUSSIAN_THERMO.items():
        found = pattern.findall(text)
        if found:
            energies[key] = float(found[-1])
    return energies


def _orca(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    if ORCA_OK not in text:
        raise SystemExit(f"{path}: ORCA did not terminate normally")
    return float(ORCA_ENERGY.findall(text)[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--program", choices=("gaussian", "orca"), required=True)
    args = parser.parse_args()
    outputs = args.dataset_dir / "outputs"
    atoms: dict[str, dict] = {}
    if args.program == "gaussian":
        for log in sorted(outputs.glob("*.log")):
            atoms[log.stem] = {"energies_hartree": _gaussian(log)}
    else:
        for out in sorted(outputs.glob("*.out")):
            element, guess = out.stem.split("_", 1)
            atoms.setdefault(element, {"energies_hartree": {}})["energies_hartree"][guess] = _orca(
                out
            )
        for atom in atoms.values():
            energies = atom["energies_hartree"]
            atom["selected_guess"] = min(energies, key=energies.get)
            atom["selected_energy_hartree"] = energies[atom["selected_guess"]]
            atom["guess_spread_hartree"] = max(energies.values()) - min(energies.values())
    if not atoms:
        raise SystemExit(f"no outputs found in {outputs}")
    (args.dataset_dir / "results.json").write_text(
        json.dumps({"program": args.program, "atoms": atoms}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
