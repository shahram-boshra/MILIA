import sys
from pathlib import Path

# Add project root to Python path (for Docker environment)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Now imports will work
import logging
from milia_pipeline.preprocessing import PreprocessorRegistry

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configure preprocessing
config = {
    'raw_tar_path': str(Path.home() / 'Chem_Data/Milia_PyG_Dataset/raw/wavefunctions.tar.gz'),
    'output_npz_path': str(Path.home() / 'Chem_Data/Milia_PyG_Dataset/raw/wavefunctions_sliced.npz'),
    'num_molecules': 10,  # Start small for testing
    'feature_tier': 'complete', # 'basic', 'standard', or 'complete'
    'cleanup_temp': True
}

# Run preprocessing
PreprocessorClass = PreprocessorRegistry.get_preprocessor("Wavefunction")
preprocessor = PreprocessorClass(config, logger)
output_path = preprocessor.run()

print(f"✓ Created: {output_path}")
