import sys
from pathlib import Path

# Add project root to Python path (for Docker environment)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Load the npz file
npz_path = Path.home() / "Chem_Data/milia_PyG_Dataset/raw/wavefunctions_sliced.npz"
data = np.load(npz_path, allow_pickle=True)

print("=" * 70)
print("NPZ FILE CONTENTS - COMPLETE TIER")
print("=" * 70)
print(f"Total keys: {len(data.keys())}")
print()

# List all keys
for i, key in enumerate(sorted(data.keys()), 1):
    if key == "metadata":
        print(f"{i:2d}. {key:30s} - metadata dict")
    else:
        print(f"{i:2d}. {key:30s} - shape: {data[key].shape}, dtype: {data[key].dtype}")

print()
print("=" * 70)
print("EXPECTED FEATURES FOR 'COMPLETE' TIER:")
print("=" * 70)

expected_features = [
    # Core features (5)
    "compounds",
    "atoms",
    "coordinates",
    "n_atoms",
    "n_electrons",
    # Basic electronic (7)
    "homo_energy_eV",
    "lumo_energy_eV",
    "homo_lumo_gap_eV",
    "homo_index",
    "lumo_index",
    "mo_energies",
    "mo_occupations",
    # Energy (2)
    "total_energy_eV",
    "total_energy_Hartree",
    # Molecular properties (2)
    "molecular_formula",
    "molecular_weight",
    # MO statistics (4)
    "mo_energy_mean_eV",
    "mo_energy_std_eV",
    "mo_energy_min_eV",
    "mo_energy_max_eV",
    # Orbital counts (2)
    "n_occupied_orbitals",
    "n_virtual_orbitals",
    # Quantum descriptors (5)
    "ionization_potential_eV",
    "electron_affinity_eV",
    "chemical_hardness_eV",
    "chemical_potential_eV",
    "electrophilicity_eV",
    # Advanced MO info (3)
    "mo_coefficients",
    "mo_kind",
    "n_basis_functions",
    # Basis info (1)
    "n_shells",
    # Metadata (1)
    "metadata",
]

print(f"Expected: {len(expected_features)} keys")
for i, feat in enumerate(expected_features, 1):
    status = "✓" if feat in data.keys() else "✗ MISSING"
    print(f"{i:2d}. {feat:30s} {status}")

print()
print("=" * 70)
print("MISSING FEATURES:")
print("=" * 70)
missing = [f for f in expected_features if f not in data.keys()]
if missing:
    for feat in missing:
        print(f"  ✗ {feat}")
else:
    print("  None - all expected features present!")

print()
print("=" * 70)
print("EXTRA FEATURES (not in expected list):")
print("=" * 70)
extra = [k for k in data.keys() if k not in expected_features]
if extra:
    for feat in extra:
        print(f"  + {feat}")
else:
    print("  None")

print()
print("=" * 70)
print(f"SUMMARY: {len(data.keys())} actual vs {len(expected_features)} expected")
print("=" * 70)
