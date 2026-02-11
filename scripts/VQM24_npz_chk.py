#!/usr/bin/env python3
"""
Inspect wavefunctions_sliced.npz - Sample 3 molecules with numerical data
"""
import numpy as np
from pathlib import Path

# Load NPZ
npz_path = Path.home() / 'Chem_Data/milia_PyG_Dataset/raw/wavefunctions_sliced.npz'
data = np.load(npz_path, allow_pickle=True)

print("=" * 80)
print("WAVEFUNCTION NPZ INSPECTION - 3 SAMPLES WITH NUMERICAL DATA")
print("=" * 80)
print(f"\nFile: {npz_path}")
print(f"Total keys: {len(data.files)}")
print(f"Total molecules: {len(data['compounds'])}")
print(f"File size: {npz_path.stat().st_size / (1024**2):.2f} MB\n")

# Inspect 3 molecules
for i in range(3):
    print("=" * 80)
    print(f"MOLECULE {i+1}: {data['compounds'][i]}")
    print("=" * 80)
    
    print(f"\n--- GEOMETRIC DATA ---")
    print(f"n_atoms: {data['n_atoms'][i]}")
    print(f"molecular_formula: {data['molecular_formula'][i]}")
    print(f"molecular_weight: {data['molecular_weight'][i]:.4f} amu")
    print(f"\natoms (atomic numbers):\n  {data['atoms'][i]}")
    print(f"\ncoordinates (Bohr) [first 3 atoms]:")
    for j in range(min(3, len(data['coordinates'][i]))):
        coord = data['coordinates'][i][j]
        print(f"  Atom {j+1}: [{coord[0]:10.6f}, {coord[1]:10.6f}, {coord[2]:10.6f}]")
    
    print(f"\n--- ELECTRONIC PROPERTIES ---")
    print(f"n_electrons: {data['n_electrons'][i]}")
    print(f"mo_kind: {data['mo_kind'][i]}")
    print(f"n_basis: {data['n_basis'][i]}")
    print(f"n_occupied_orbitals: {data['n_occupied_orbitals'][i]}")
    print(f"n_virtual_orbitals: {data['n_virtual_orbitals'][i]}")
    
    print(f"\n--- FRONTIER ORBITAL ENERGIES (Hartree) ---")
    print(f"homo_energy: {data['homo_energy'][i]:.6f}")
    print(f"lumo_energy: {data['lumo_energy'][i]:.6f}")
    print(f"homo_lumo_gap: {data['homo_lumo_gap'][i]:.6f} Ha ({data['homo_lumo_gap_eV'][i]:.4f} eV)")
    
    print(f"\n--- ORBITAL ENERGY STATISTICS (Hartree) ---")
    print(f"Occupied orbitals:")
    print(f"  mean: {data['occupied_energy_mean'][i]:10.6f}")
    print(f"  std:  {data['occupied_energy_std'][i]:10.6f}")
    print(f"  min:  {data['occupied_energy_min'][i]:10.6f}")
    print(f"  max:  {data['occupied_energy_max'][i]:10.6f}")
    print(f"Virtual orbitals:")
    print(f"  mean: {data['virtual_energy_mean'][i]:10.6f}")
    print(f"  std:  {data['virtual_energy_std'][i]:10.6f}")
    print(f"  min:  {data['virtual_energy_min'][i]:10.6f}")
    print(f"  max:  {data['virtual_energy_max'][i]:10.6f}")
    
    print(f"\n--- CHEMICAL DESCRIPTORS (Hartree) ---")
    print(f"ionization_potential_approx: {data['ionization_potential_approx'][i]:.6f}")
    print(f"electron_affinity_approx: {data['electron_affinity_approx'][i]:.6f}")
    print(f"chemical_hardness: {data['chemical_hardness'][i]:.6f}")
    print(f"chemical_potential: {data['chemical_potential'][i]:.6f}")
    
    print(f"\n--- MO ENERGIES (first 5 and last 5, Hartree) ---")
    mo_e = data['mo_energies'][i]
    print(f"First 5: {mo_e[:5]}")
    print(f"Last 5:  {mo_e[-5:]}")
    
    print(f"\n--- MO OCCUPATIONS (first 10 and last 10) ---")
    mo_occ = data['mo_occupations'][i]
    print(f"First 10: {mo_occ[:10]}")
    print(f"Last 10:  {mo_occ[-10:]}")
    
    print(f"\n--- MO COEFFICIENTS (sample: first 3x3 block) ---")
    mo_coef = data['mo_coefficients'][i]
    print(f"Shape: {mo_coef.shape}")
    for j in range(3):
        print(f"  {mo_coef[j, :3]}")
    
    print(f"\n--- DATA TYPES ---")
    print(f"atoms: {data['atoms'][i].dtype}")
    print(f"coordinates: {data['coordinates'][i].dtype}")
    print(f"mo_energies: {data['mo_energies'][i].dtype}")
    print(f"mo_occupations: {data['mo_occupations'][i].dtype}")
    print(f"mo_coefficients: {data['mo_coefficients'][i].dtype}")
    print()

print("=" * 80)
print("COMPLETE KEY LIST")
print("=" * 80)
for idx, key in enumerate(data.files, 1):
    if key == 'metadata':
        continue
    dtype = str(data[key].dtype)
    shape = data[key].shape
    print(f"{idx:2d}. {key:30s} dtype={dtype:10s} shape={shape}")

print("\n" + "=" * 80)
print("METADATA")
print("=" * 80)
metadata = data['metadata'][0]
for k, v in metadata.items():
    print(f"  {k}: {v}")
print("=" * 80)
