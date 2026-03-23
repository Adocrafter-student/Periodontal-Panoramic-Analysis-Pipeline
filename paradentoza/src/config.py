from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"

BINARY_HEALTHY = DATASET_ROOT / "dental-panoramic" / "penyakit-non-periodontal"
BINARY_DISEASE = DATASET_ROOT / "dental-panoramic" / "penyakit-periodontal"

STAGE_DIRS = [
    DATASET_ROOT / "stage_1",
    DATASET_ROOT / "stage_2",
    DATASET_ROOT / "stage_3",
    DATASET_ROOT / "stage_4",
]

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
