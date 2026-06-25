# milia_pipeline/preprocessing/preprocessors/qm40.py

"""
QM40 Preprocessor
=================

Preprocessor for the QM40 quantum chemistry dataset (ZIP archive of CSV files).

Extracts ``QM40_dataset.zip``, parses the three QM40 CSV files
(``QM40_main.csv``, ``QM40_xyz.csv``, ``QM40_bond.csv``) via the QM40 CSV
parser, and writes a unified ``.npz`` file compatible with miliaDataset using
the shared ``npz_builders.build_npz`` utility.

QM40 Dataset Information:
-------------------------
- Reference: Madushanka, Moura Jr. & Kraka. Sci Data 11, 1376 (2024).
- DOI: 10.1038/s41597-024-04206-y
- Figshare DOI: 10.6084/m9.figshare.25993060
- Download URL: https://ndownloader.figshare.com/files/47535647
- File: QM40_dataset.zip
- Contents: 162,954 drug-like ZINC molecules, B3LYP/6-31G(2df,p) (Gaussian16).
- Charge state: NEUTRAL only (anions and cations are excluded; charge-neutral
  singlet ground states), so no molecular-charge tracking is required.
- Elements: C, O, N, S, F, Cl (+ H).

Archive Structure (verified from the real archive):
----------------------------------------------------
QM40_dataset.zip
├── QM40 dataset/                 ← note the SPACE in the folder name
│   ├── QM40_main.csv             ← one row per molecule (16 QM params)
│   ├── QM40_xyz.csv              ← one row per atom (symbol, init/final xyz, Mulliken)
│   └── QM40_bond.csv             ← one row per bond (1-based indices, tag, local-mode Ka)
└── __MACOSX/                     ← macOS AppleDouble junk (ignored by the parser)

Unlike xxMD there are NO nested ZIP files; a single top-level extraction is
sufficient. CSV discovery, ``__MACOSX``/``._*`` filtering, the join-by-``Zinc_id``
logic, and the 1-based -> 0-based bond-index conversion all live in
``qm40_csv_parser.parse_qm40_csv_files``.

NPZ construction uses the shared ``npz_builders.build_npz`` (which enforces the
required ``compounds``/``atoms``/``coordinates`` keys and appends the
``metadata`` key expected by ``BasePreprocessor._validate_output``).

Author: milia Pipeline Team
Version: 1.0.0
"""

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

from milia_pipeline.exceptions import ConfigurationError, DataProcessingError
from milia_pipeline.preprocessing.base_preprocessor import BasePreprocessor
from milia_pipeline.preprocessing.registry import PreprocessorRegistry
from milia_pipeline.preprocessing.utils.npz_builders import build_npz
from milia_pipeline.preprocessing.utils.qm40_csv_parser import parse_qm40_csv_files

logger = logging.getLogger(__name__)


@PreprocessorRegistry.register("QM40")
class QM40Preprocessor(BasePreprocessor):
    """
    Preprocessor for the QM40 dataset.

    Pipeline:
    ---------
    1. Extract the ZIP archive to a temporary directory.
    2. Parse the three QM40 CSV files and join them by ``Zinc_id``.
    3. Build a unified ``.npz`` file via the shared ``build_npz`` utility.
    4. Clean up the temporary extraction directory.

    Configuration:
    --------------
    Required keys:
        - raw_archive_path: Path to QM40_dataset.zip
        - output_npz_path: Path for the output .npz file

    Optional keys:
        - num_molecules: Limit number of molecules (None = all)
        - include_bond_data: Store per-bond local-mode force constants (default: True)
        - include_initial_coordinates: Store pre-optimization coords (default: True)
        - cleanup_temp: Remove the temporary extraction dir afterward (default: True)
    """

    def _validate_config(self) -> None:
        """Validate QM40-specific configuration requirements."""
        required_keys = ["raw_archive_path", "output_npz_path"]

        for key in required_keys:
            if key not in self.config:
                raise ConfigurationError(
                    f"QM40 preprocessor missing required config key: {key}", config_key=key
                )

        # Validate archive exists
        archive_path = Path(self.config["raw_archive_path"])
        if not archive_path.exists():
            raise ConfigurationError(
                f"QM40 archive not found: {archive_path}", config_key="raw_archive_path"
            )

        # Validate archive is a ZIP file
        if not archive_path.name.endswith(".zip"):
            self.logger.warning(f"QM40 archive does not have .zip extension: {archive_path.name}")

    def preprocess(self) -> Path:
        """
        Main preprocessing pipeline for the QM40 dataset.

        Returns:
            Path to the created .npz file.

        Raises:
            DataProcessingError: If preprocessing fails.
        """
        try:
            archive_path = Path(self.config["raw_archive_path"])
            output_npz = Path(self.config["output_npz_path"])

            # Optional configuration (flat keys; the CLI flattens
            # processing_config.preprocessing to the root before passing here)
            num_molecules = self.config.get("num_molecules", None)
            include_bond_data = self.config.get("include_bond_data", True)
            include_initial_coordinates = self.config.get("include_initial_coordinates", True)
            cleanup_temp = self.config.get("cleanup_temp", True)

            self.logger.info("=" * 60)
            self.logger.info("QM40 Preprocessing Pipeline")
            self.logger.info("=" * 60)
            self.logger.info(f"Archive: {archive_path}")
            self.logger.info(f"Output: {output_npz}")
            self.logger.info(f"Molecule limit: {num_molecules if num_molecules else 'all'}")
            self.logger.info(f"Include bond data: {include_bond_data}")
            self.logger.info(f"Include initial coordinates: {include_initial_coordinates}")
            self.logger.info("=" * 60)

            # Check if output already exists
            if output_npz.exists():
                size_mb = output_npz.stat().st_size / (1024**2)
                self.logger.info(f"Found existing {output_npz.name} ({size_mb:.2f} MB)")
                self.logger.info("Skipping preprocessing - delete the file to regenerate")
                return output_npz

            # Step 1: Extract archive
            self.logger.info("Step 1: Extracting ZIP archive...")
            extracted_dir = self._extract_archive(archive_path)

            try:
                # Step 2: Parse CSV files (join by Zinc_id handled in the parser)
                self.logger.info("Step 2: Parsing QM40 CSV files...")
                features, metadata = parse_qm40_csv_files(
                    csv_dir=extracted_dir,
                    max_molecules=num_molecules,
                    include_bond_data=include_bond_data,
                    include_initial_coordinates=include_initial_coordinates,
                    logger=self.logger,
                )

                # Enrich metadata with provenance (build_npz adds num_molecules/feature_keys)
                metadata.update(
                    {
                        "version": "1.0.0",
                        "dataset_name": "QM40",
                        "source": archive_path.name,
                        "doi": "10.1038/s41597-024-04206-y",
                        "figshare_doi": "10.6084/m9.figshare.25993060",
                    }
                )

                # Step 3: Build unified NPZ file via the shared utility
                self.logger.info("Step 3: Building NPZ file...")
                build_npz(features, metadata, output_npz, self.logger)

            finally:
                # Step 4: Cleanup
                if cleanup_temp and extracted_dir.exists():
                    self.logger.info("Step 4: Cleaning up temporary files...")
                    shutil.rmtree(extracted_dir)
                    self.logger.info(f"  Removed: {extracted_dir}")

            self.logger.info("=" * 60)
            self.logger.info("QM40 Preprocessing Complete!")
            self.logger.info("=" * 60)

            return output_npz

        except Exception as e:
            raise DataProcessingError(
                f"QM40 preprocessing failed: {e}", operation="qm40_preprocessing"
            ) from e

    def _extract_archive(self, archive_path: Path) -> Path:
        """
        Extract the QM40 ZIP archive to a temporary directory.

        QM40 has a single top-level archive (no nested ZIPs). CSV discovery and
        macOS-junk (``__MACOSX``/``._*``) filtering are performed downstream by
        ``parse_qm40_csv_files``, so this method only needs to extract and return
        the extraction root.

        Args:
            archive_path: Path to QM40_dataset.zip.

        Returns:
            Path to the temporary extraction directory.

        Raises:
            DataProcessingError: If the archive cannot be extracted.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="qm40_extract_"))
        self.logger.info(f"  Extracting to: {temp_dir}")

        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                members = zf.namelist()
                self.logger.info(f"  Archive contains {len(members)} item(s)")
                zf.extractall(temp_dir)

            return temp_dir

        except zipfile.BadZipFile as e:
            # Cleanup on failure
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise DataProcessingError(
                f"Failed to extract QM40 archive: {e}",
                file_path=str(archive_path),
                operation="archive_extraction",
            ) from e
