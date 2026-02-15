#!/usr/bin/env python3
"""
Test Dataset Generator for Preprocessing Tests
===============================================

Helper script to generate small test .tar.gz files containing .molden files
for testing the preprocessing subsystem.

Features:
---------
- Generate realistic .molden file structures
- Create tar.gz archives with configurable molecule counts
- Support for multiple complexity levels
- Validation of generated files

Usage:
------
As a script:
    python create_test_dataset.py --num-molecules 10 --output test_data.tar.gz

As a module:
    from create_test_dataset import create_test_dataset
    create_test_dataset(num_molecules=10, output_path='test_data.tar.gz')

NOTE: This script runs inside Docker at /app/milia

Author: Milia Pipeline Team
Version: 1.0
Date: November 2025
"""

import sys
from pathlib import Path

# CRITICAL: Add project root to Python path FIRST (Docker-compatible)
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import argparse
import logging
import random
import tarfile
import tempfile

# ==========================================
# LOGGING SETUP
# ==========================================

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# ==========================================
# MOLDEN FILE TEMPLATES
# ==========================================


def generate_molden_content(
    molecule_name: str, num_atoms: int = 2, complexity: str = "simple"
) -> str:
    """
    Generate realistic .molden file content.

    Args:
        molecule_name: Name/identifier for the molecule
        num_atoms: Number of atoms in the molecule
        complexity: Complexity level ('simple', 'standard', 'complex')

    Returns:
        Complete .molden file content as string
    """
    if complexity == "simple":
        return _generate_simple_molden(molecule_name, num_atoms)
    elif complexity == "standard":
        return _generate_standard_molden(molecule_name, num_atoms)
    elif complexity == "complex":
        return _generate_complex_molden(molecule_name, num_atoms)
    else:
        raise ValueError(f"Unknown complexity: {complexity}")


def _generate_simple_molden(molecule_name: str, num_atoms: int) -> str:
    """Generate simple H2-like molecule with minimal data."""
    content = ["[Molden Format]"]
    content.append(f"[Title]\n{molecule_name}\n")

    # Atoms section (Hydrogen atoms for simplicity)
    content.append("[Atoms] AU")
    for i in range(num_atoms):
        z = random.uniform(-2.0, 2.0)
        content.append(f"H  1  1  0.0  0.0  {z:.6f}")

    # GTO section (minimal basis)
    content.append("\n[GTO]")
    for i in range(1, num_atoms + 1):
        content.append(f"{i} 0")
        content.append("s 1 1.0")
        content.append("1.0 1.0")
        content.append("")

    # MO section (minimal orbitals)
    content.append("[MO]")

    # HOMO (occupied)
    content.append(f"Ene= {random.uniform(-1.0, -0.3):.6f}")
    content.append("Spin= Alpha")
    content.append("Occup= 1.0")
    for i in range(1, num_atoms + 1):
        content.append(f"{i} {random.uniform(0.5, 1.0):.6f}")

    # LUMO (unoccupied)
    content.append(f"Ene= {random.uniform(0.3, 1.0):.6f}")
    content.append("Spin= Alpha")
    content.append("Occup= 0.0")
    for i in range(1, num_atoms + 1):
        content.append(f"{i} {random.uniform(0.0, 0.5):.6f}")

    return "\n".join(content)


def _generate_standard_molden(molecule_name: str, num_atoms: int) -> str:
    """Generate standard molecule with typical quantum chemistry data."""
    content = ["[Molden Format]"]
    content.append(f"[Title]\n{molecule_name}\n")

    # Atoms section (mix of H, C, N, O)
    atom_types = ["H", "C", "N", "O"]
    atom_numbers = {"H": 1, "C": 6, "N": 7, "O": 8}

    content.append("[Atoms] AU")
    selected_atoms = []
    for i in range(num_atoms):
        atom = random.choice(atom_types)
        selected_atoms.append(atom)
        x = random.uniform(-3.0, 3.0)
        y = random.uniform(-3.0, 3.0)
        z = random.uniform(-3.0, 3.0)
        content.append(f"{atom}  {i + 1}  {atom_numbers[atom]}  {x:.6f}  {y:.6f}  {z:.6f}")

    # GTO section (STO-3G basis)
    content.append("\n[GTO]")
    for i in range(1, num_atoms + 1):
        content.append(f"{i} 0")

        # s orbital
        content.append("s 3 1.0")
        content.append("0.1688554040 0.4441530000")
        content.append("0.6239137298 0.5353281423")
        content.append("3.4252509140 0.1543289673")

        # p orbital (if not hydrogen)
        if selected_atoms[i - 1] != "H":
            content.append("p 3 1.0")
            content.append("0.1688554040 0.4441530000")
            content.append("0.6239137298 0.5353281423")
            content.append("3.4252509140 0.1543289673")

        content.append("")

    # MO section (more orbitals)
    content.append("[MO]")

    num_orbitals = num_atoms * 2  # Approximate number of MOs
    num_electrons = sum(atom_numbers[atom] for atom in selected_atoms)

    for mo_idx in range(num_orbitals):
        # Energy increases with orbital index
        if mo_idx < num_electrons // 2:
            # Occupied orbital
            energy = random.uniform(-2.0, -0.2)
            occup = 2.0 if mo_idx < num_electrons // 2 - 1 else float(num_electrons % 2)
        else:
            # Virtual orbital
            energy = random.uniform(0.1, 3.0)
            occup = 0.0

        content.append(f"Ene= {energy:.6f}")
        content.append("Spin= Alpha")
        content.append(f"Occup= {occup:.1f}")

        for i in range(1, num_atoms + 1):
            content.append(f"{i} {random.uniform(-1.0, 1.0):.6f}")

    return "\n".join(content)


def _generate_complex_molden(molecule_name: str, num_atoms: int) -> str:
    """Generate complex molecule with extensive quantum chemistry data."""
    # Start with standard content
    content_list = _generate_standard_molden(molecule_name, num_atoms).split("\n")

    # Add additional sections for complex tier

    # Add 5D section
    content_list.append("\n[5D]")

    # Add 7F section
    content_list.append("[7F]")

    # Add charge and spin multiplicity
    content_list.insert(2, "[CHARGE]\n0")
    content_list.insert(3, "[SPIN]\n1")

    return "\n".join(content_list)


# ==========================================
# DATASET GENERATION
# ==========================================


def create_test_dataset(
    num_molecules: int,
    output_path: Path,
    complexity: str = "simple",
    atoms_per_molecule: tuple[int, int] | None = None,
    compression: str = "gz",
) -> Path:
    """
    Create a test .tar.gz dataset with .molden files.

    Args:
        num_molecules: Number of molecules to generate
        output_path: Path for output .tar.gz file
        complexity: Complexity level ('simple', 'standard', 'complex')
        atoms_per_molecule: (min, max) range for atoms per molecule
        compression: Compression type ('gz', 'bz2', or '' for no compression)

    Returns:
        Path to created dataset file

    Raises:
        ValueError: If parameters are invalid
    """
    if num_molecules < 1:
        raise ValueError(f"num_molecules must be >= 1, got {num_molecules}")

    if atoms_per_molecule is None:
        atoms_per_molecule = (2, 10)

    min_atoms, max_atoms = atoms_per_molecule
    if min_atoms < 1 or max_atoms < min_atoms:
        raise ValueError(f"Invalid atoms_per_molecule range: {atoms_per_molecule}")

    logger.info(f"Creating test dataset: {num_molecules} molecules")
    logger.info(f"Output: {output_path}")
    logger.info(f"Complexity: {complexity}")
    logger.info(f"Atoms per molecule: {min_atoms}-{max_atoms}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for .molden files
    with tempfile.TemporaryDirectory(prefix="molden_temp_") as temp_dir:
        temp_path = Path(temp_dir)

        # Generate .molden files
        molden_files = []
        for i in range(num_molecules):
            molecule_name = f"molecule_{i:06d}"
            num_atoms = random.randint(min_atoms, max_atoms)

            # Generate content
            content = generate_molden_content(
                molecule_name=molecule_name, num_atoms=num_atoms, complexity=complexity
            )

            # Write to file
            molden_file = temp_path / f"{molecule_name}.molden"
            molden_file.write_text(content)
            molden_files.append(molden_file)

            if (i + 1) % 10 == 0 or i == num_molecules - 1:
                logger.info(f"Generated {i + 1}/{num_molecules} molecules")

        # Create tar archive
        logger.info(f"Creating tar.{compression} archive...")

        tar_mode = "w" if not compression else f"w:{compression}"
        with tarfile.open(output_path, tar_mode) as tar:
            for molden_file in molden_files:
                tar.add(molden_file, arcname=molden_file.name)

        logger.info(f"✓ Created {output_path}")

        # Report file size
        file_size = output_path.stat().st_size
        if file_size < 1024:
            size_str = f"{file_size} bytes"
        elif file_size < 1024**2:
            size_str = f"{file_size / 1024:.2f} KB"
        else:
            size_str = f"{file_size / (1024**2):.2f} MB"

        logger.info(f"  File size: {size_str}")
        logger.info(f"  Molecules: {num_molecules}")
        logger.info(f"  Complexity: {complexity}")

    return output_path


def validate_test_dataset(dataset_path: Path) -> dict:
    """
    Validate a test dataset file.

    Args:
        dataset_path: Path to .tar.gz file

    Returns:
        Dictionary with validation results
    """
    if not dataset_path.exists():
        return {"valid": False, "error": f"File not found: {dataset_path}"}

    try:
        with tarfile.open(dataset_path, "r:*") as tar:
            members = tar.getmembers()
            molden_files = [m for m in members if m.name.endswith(".molden")]

            # Extract and validate first file
            if molden_files:
                first_file = tar.extractfile(molden_files[0])
                content = first_file.read().decode("utf-8")
                has_molden_format = "[Molden Format]" in content
            else:
                has_molden_format = False

            return {
                "valid": True,
                "total_files": len(members),
                "molden_files": len(molden_files),
                "has_molden_format": has_molden_format,
                "file_size": dataset_path.stat().st_size,
                "compression": dataset_path.suffix,
            }

    except Exception as e:
        return {"valid": False, "error": str(e)}


# ==========================================
# PRESET DATASETS
# ==========================================


def create_small_test_dataset(output_dir: Path) -> Path:
    """Create small test dataset (5 molecules, simple)."""
    output_path = output_dir / "small_test_dataset.tar.gz"
    return create_test_dataset(
        num_molecules=5, output_path=output_path, complexity="simple", atoms_per_molecule=(2, 5)
    )


def create_medium_test_dataset(output_dir: Path) -> Path:
    """Create medium test dataset (20 molecules, standard)."""
    output_path = output_dir / "medium_test_dataset.tar.gz"
    return create_test_dataset(
        num_molecules=20, output_path=output_path, complexity="standard", atoms_per_molecule=(3, 10)
    )


def create_large_test_dataset(output_dir: Path) -> Path:
    """Create large test dataset (100 molecules, complex)."""
    output_path = output_dir / "large_test_dataset.tar.gz"
    return create_test_dataset(
        num_molecules=100, output_path=output_path, complexity="complex", atoms_per_molecule=(5, 15)
    )


def create_all_preset_datasets(output_dir: Path) -> list[Path]:
    """Create all preset test datasets."""
    logger.info("Creating all preset test datasets...")

    datasets = []

    # Small
    logger.info("\n" + "=" * 70)
    logger.info("SMALL DATASET")
    logger.info("=" * 70)
    datasets.append(create_small_test_dataset(output_dir))

    # Medium
    logger.info("\n" + "=" * 70)
    logger.info("MEDIUM DATASET")
    logger.info("=" * 70)
    datasets.append(create_medium_test_dataset(output_dir))

    # Large
    logger.info("\n" + "=" * 70)
    logger.info("LARGE DATASET")
    logger.info("=" * 70)
    datasets.append(create_large_test_dataset(output_dir))

    logger.info("\n" + "=" * 70)
    logger.info("ALL DATASETS CREATED")
    logger.info("=" * 70)

    for dataset in datasets:
        logger.info(f"  ✓ {dataset.name}")

    return datasets


# ==========================================
# COMMAND LINE INTERFACE
# ==========================================


def main():
    """Command-line interface for test dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate test datasets for preprocessing tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create small test dataset
  python create_test_dataset.py --preset small --output-dir ./test_data

  # Create custom dataset
  python create_test_dataset.py --num-molecules 50 --complexity standard \\
      --output ./test_data/custom.tar.gz

  # Create all preset datasets
  python create_test_dataset.py --preset all --output-dir ./test_data

  # Validate existing dataset
  python create_test_dataset.py --validate ./test_data/dataset.tar.gz
        """,
    )

    parser.add_argument("--num-molecules", type=int, help="Number of molecules to generate")

    parser.add_argument("--output", type=Path, help="Output path for .tar.gz file")

    parser.add_argument(
        "--complexity",
        choices=["simple", "standard", "complex"],
        default="simple",
        help="Complexity level (default: simple)",
    )

    parser.add_argument(
        "--atoms-min", type=int, default=2, help="Minimum atoms per molecule (default: 2)"
    )

    parser.add_argument(
        "--atoms-max", type=int, default=10, help="Maximum atoms per molecule (default: 10)"
    )

    parser.add_argument(
        "--preset", choices=["small", "medium", "large", "all"], help="Create preset dataset"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_data"),
        help="Output directory for preset datasets (default: test_data)",
    )

    parser.add_argument("--validate", type=Path, help="Validate existing dataset file")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validation mode
    if args.validate:
        logger.info(f"Validating dataset: {args.validate}")
        result = validate_test_dataset(args.validate)

        if result["valid"]:
            logger.info("✓ Dataset is valid")
            logger.info(f"  Total files: {result['total_files']}")
            logger.info(f"  .molden files: {result['molden_files']}")
            logger.info(f"  Has Molden Format: {result['has_molden_format']}")
            logger.info(f"  File size: {result['file_size'] / (1024**2):.2f} MB")
            return 0
        else:
            logger.error(f"✗ Dataset is invalid: {result['error']}")
            return 1

    # Preset mode
    if args.preset:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.preset == "small":
            create_small_test_dataset(output_dir)
        elif args.preset == "medium":
            create_medium_test_dataset(output_dir)
        elif args.preset == "large":
            create_large_test_dataset(output_dir)
        elif args.preset == "all":
            create_all_preset_datasets(output_dir)

        return 0

    # Custom dataset mode
    if args.num_molecules and args.output:
        create_test_dataset(
            num_molecules=args.num_molecules,
            output_path=args.output,
            complexity=args.complexity,
            atoms_per_molecule=(args.atoms_min, args.atoms_max),
        )
        return 0

    # No valid mode specified
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
