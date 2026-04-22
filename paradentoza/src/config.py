from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"

# Binary classification (healthy vs periodontitis) — used for training
BINARY_HEALTHY = DATASET_ROOT / "dental-panoramic" / "penyakit-non-periodontal"
BINARY_DISEASE = DATASET_ROOT / "dental-panoramic" / "penyakit-periodontal"

# Legacy stage folders — kept for backward compat but no longer used for training
STAGE_DIRS = [
    DATASET_ROOT / "stage_1",
    DATASET_ROOT / "stage_2",
    DATASET_ROOT / "stage_3",
    DATASET_ROOT / "stage_4",
]

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Third molars excluded from staging analysis (FDI notation)
THIRD_MOLARS_FDI = frozenset({18, 28, 38, 48})
