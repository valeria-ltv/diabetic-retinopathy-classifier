from pathlib import Path

# Determine the repository root regardless of where the code is run
ROOT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "aptos2019"
SPLITS_DIR = PROCESSED_DIR / "splits"

KAGGLE_DIR = Path("/kaggle/input/competitions/aptos2019-blindness-detection")

if KAGGLE_DIR.exists():
    RAW_DIR = KAGGLE_DIR
else:
    RAW_DIR = ROOT_DIR / "data" / "raw" / "aptos2019"

IMG_DIR = RAW_DIR / "train_images"