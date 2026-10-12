from pathlib import Path

# Determine the repository root regardless of where the code is run
ROOT_DIR = Path(__file__).resolve().parent.parent

# Determining the environment (kaggle or local)
KAGGLE_TRAIN_DIR = Path("/kaggle/input/competitions/aptos2019-blindness-detection/train_images")

if KAGGLE_TRAIN_DIR.exists():
    # Setup for kaggle
    IMG_DIR = KAGGLE_TRAIN_DIR
else:
    # Setup for local
    IMG_DIR = ROOT_DIR / "data" / "raw" / "aptos2019" / "train_images"

CSV_PATH = ROOT_DIR / "data" / "processed" / "aptos2019" / "splits" / "train_split.csv"