#!/usr/bin/env python3
"""
milia Wavefunction Dataset - Molden File Parser Test
Uses IOData to extract molecular features from .molden files
"""

import sys
from pathlib import Path
import numpy as np

print("=" * 80)
print("milia WAVEFUNCTION DATASET - MOLDEN FILE PARSER TEST")
print("=" * 80)

# Import IOData
try:
    from iodata import load_one
    print("✓ IOData imported successfully")
except ImportError as e:
    print(f"✗ Failed to import IOData: {e}")
    sys.exit(1)

# Import PyTorch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} imported successfully")
except ImportError as e:
    print(f"✗ Failed to import PyTorch: {e}")
    sys.exit(1)

print("=" * 80)

# Define path to wavefunction files
data_dir = Path("/root/Chem_Data/milia_PyG_Dataset/raw/wavefunction_sliced")

# Check if directory exists
if not data_dir.exists():
    print(f"✗ ERROR: Directory not found: {data_dir}")
    sys.exit(1)

print(f"✓ Data directory found: {data_dir}")

# Find all .molden files
molden_files = list(data_dir.glob("*.molden"))
print(f"✓ Found {len(molden_files)} .molden files")

if len(molden_files) == 0:
    print("✗ ERROR: No .molden files found!")
    sys.exit(1)

# Select first file for testing
test_file = molden_files[0]
print(f"\n{'=' * 80}")
print(f"TESTING WITH FILE: {test_file.name}")
print(f"{'=' * 80}\n")

# Parse the .molden file
print("Parsing .molden file with IOData...")
try:
    mol_data = load_one(str(test_file))
    print("✓ File parsed successfully!")
except Exception as e:
    print(f"✗ ERROR parsing file: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("MOLECULAR STRUCTURE INFORMATION")
print("=" * 80)

# Basic molecular info
print(f"\nNumber of atoms: {len(mol_data.atnums)}")
print(f"Atomic numbers: {mol_data.atnums}")

# Get atom types
from collections import Counter
atom_types = Counter(mol_data.atnums)
print(f"\nAtom composition:")
# Simple element lookup (extend as needed)
element_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 14: 'Si', 16: 'S'}
for atnum, count in sorted(atom_types.items()):
    element = element_map.get(atnum, f'Z={atnum}')
    print(f"  {element}: {count}")

# Molecular formula
formula_parts = []
for atnum, count in sorted(atom_types.items(), key=lambda x: -x[0]):
    element = element_map.get(atnum, f'Z={atnum}')
    if count > 1:
        formula_parts.append(f"{element}{count}")
    else:
        formula_parts.append(element)
molecular_formula = ''.join(formula_parts)
print(f"\nMolecular formula: {molecular_formula}")

# Coordinates
print(f"\nCoordinates shape: {mol_data.atcoords.shape}")
print(f"Coordinate units: Bohr (atomic units)")
print(f"Coordinate range:")
print(f"  X: [{mol_data.atcoords[:, 0].min():.4f}, {mol_data.atcoords[:, 0].max():.4f}]")
print(f"  Y: [{mol_data.atcoords[:, 1].min():.4f}, {mol_data.atcoords[:, 1].max():.4f}]")
print(f"  Z: [{mol_data.atcoords[:, 2].min():.4f}, {mol_data.atcoords[:, 2].max():.4f}]")

print("\n" + "=" * 80)
print("BASIS SET INFORMATION")
print("=" * 80)

if hasattr(mol_data, 'obasis') and mol_data.obasis is not None:
    print(f"\nNumber of basis functions: {mol_data.obasis.nbasis}")
    print(f"Number of shells: {len(mol_data.obasis.shells)}")
    print(f"Basis set conventions: {mol_data.obasis.conventions}")
    print(f"Primitive normalization: {mol_data.obasis.primitive_normalization}")
else:
    print("\n✗ No basis set information available")

print("\n" + "=" * 80)
print("MOLECULAR ORBITAL INFORMATION")
print("=" * 80)

if hasattr(mol_data, 'mo') and mol_data.mo is not None:
    print(f"\nMO kind: {mol_data.mo.kind}")
    print(f"Number of MOs: {len(mol_data.mo.energies)}")
    print(f"MO coefficients shape: {mol_data.mo.coeffs.shape}")
    
    # Orbital energies
    print(f"\nOrbital energies (Hartree):")
    print(f"  Min: {mol_data.mo.energies.min():.6f}")
    print(f"  Max: {mol_data.mo.energies.max():.6f}")
    print(f"  Range: {mol_data.mo.energies.max() - mol_data.mo.energies.min():.6f}")
    
    # Occupations
    print(f"\nOrbital occupations:")
    print(f"  Min: {mol_data.mo.occs.min():.6f}")
    print(f"  Max: {mol_data.mo.occs.max():.6f}")
    
    # Count occupied vs virtual
    occupied = (mol_data.mo.occs > 0.1).sum()
    virtual = (mol_data.mo.occs < 0.1).sum()
    print(f"  Occupied orbitals: {occupied}")
    print(f"  Virtual orbitals: {virtual}")
    
    # Total electrons
    n_electrons = mol_data.mo.occs.sum()
    print(f"\nTotal electrons: {n_electrons:.1f}")
    
    # HOMO/LUMO analysis
    occupied_mask = mol_data.mo.occs > 0.1
    if occupied_mask.any():
        homo_energy = mol_data.mo.energies[occupied_mask].max()
        homo_idx = np.where(mol_data.mo.energies == homo_energy)[0][0]
        print(f"\nHOMO (orbital {homo_idx + 1}):")
        print(f"  Energy: {homo_energy:.6f} Hartree")
        print(f"  Occupation: {mol_data.mo.occs[homo_idx]:.6f}")
        
        virtual_mask = mol_data.mo.occs < 0.1
        if virtual_mask.any():
            lumo_energy = mol_data.mo.energies[virtual_mask].min()
            lumo_idx = np.where(mol_data.mo.energies == lumo_energy)[0][0]
            print(f"\nLUMO (orbital {lumo_idx + 1}):")
            print(f"  Energy: {lumo_energy:.6f} Hartree")
            print(f"  Occupation: {mol_data.mo.occs[lumo_idx]:.6f}")
            
            # HOMO-LUMO gap
            gap = lumo_energy - homo_energy
            gap_ev = gap * 27.211386  # Convert Hartree to eV
            print(f"\nHOMO-LUMO Gap:")
            print(f"  {gap:.6f} Hartree")
            print(f"  {gap_ev:.6f} eV")
else:
    print("\n✗ No molecular orbital information available")

print("\n" + "=" * 80)
print("PYTORCH TENSOR CONVERSION")
print("=" * 80)

print("\nConverting molecular data to PyTorch tensors...")

# Convert atomic numbers
atomic_numbers_torch = torch.from_numpy(mol_data.atnums.astype(np.int64))
print(f"\n✓ Atomic numbers tensor: {atomic_numbers_torch.shape}, dtype={atomic_numbers_torch.dtype}")

# Convert coordinates
coordinates_torch = torch.from_numpy(mol_data.atcoords.astype(np.float32))
print(f"✓ Coordinates tensor: {coordinates_torch.shape}, dtype={coordinates_torch.dtype}")

# Convert MO data if available
if hasattr(mol_data, 'mo') and mol_data.mo is not None:
    mo_energies_torch = torch.from_numpy(mol_data.mo.energies.astype(np.float32))
    print(f"✓ MO energies tensor: {mo_energies_torch.shape}, dtype={mo_energies_torch.dtype}")
    
    mo_occs_torch = torch.from_numpy(mol_data.mo.occs.astype(np.float32))
    print(f"✓ MO occupations tensor: {mo_occs_torch.shape}, dtype={mo_occs_torch.dtype}")
    
    mo_coeffs_torch = torch.from_numpy(mol_data.mo.coeffs.astype(np.float32))
    print(f"✓ MO coefficients tensor: {mo_coeffs_torch.shape}, dtype={mo_coeffs_torch.dtype}")
    print(f"  Memory: {mo_coeffs_torch.element_size() * mo_coeffs_torch.nelement() / 1024:.2f} KB")

print("\n" + "=" * 80)
print("FEATURE EXTRACTION SUMMARY")
print("=" * 80)

print("\nAvailable features for ML/DL:")
print("  ✓ Atomic numbers (graph nodes)")
print("  ✓ 3D coordinates (spatial information)")
print("  ✓ Molecular formula (composition)")
if hasattr(mol_data, 'obasis') and mol_data.obasis is not None:
    print(f"  ✓ Basis set size: {mol_data.obasis.nbasis} functions")
if hasattr(mol_data, 'mo') and mol_data.mo is not None:
    print(f"  ✓ MO energies: {len(mol_data.mo.energies)} orbitals")
    print(f"  ✓ MO coefficients: Full wavefunction")
    print(f"  ✓ Electron count: {mol_data.mo.occs.sum():.1f}")
    if occupied_mask.any() and virtual_mask.any():
        print(f"  ✓ HOMO-LUMO gap: {gap_ev:.3f} eV")

print("\n" + "=" * 80)
print("PYTORCH GEOMETRIC COMPATIBILITY")
print("=" * 80)

print("\nData can be converted to PyG Data object:")
print("  - Node features (x): atomic_numbers or one-hot encoding")
print("  - Node positions (pos): 3D coordinates")
print("  - Edge index: Can be built from distance cutoff or bonding")
print("  - Global features: MO energies, HOMO-LUMO gap, etc.")

print("\n" + "=" * 80)
print("TEST COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nParsed file: {test_file.name}")
print(f"Total .molden files available: {len(molden_files)}")
print("\nIOData successfully extracts:")
print("  ✓ Molecular geometry")
print("  ✓ Basis set information")
print("  ✓ Molecular orbitals")
print("  ✓ Electronic structure properties")
print("\nReady for milia Wavefunction Handler implementation!")
print("=" * 80)
