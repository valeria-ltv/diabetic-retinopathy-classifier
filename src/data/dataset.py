import warnings
from collections.abc import Callable
from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import IMG_DIR, SPLITS_DIR, PROCESSED_DIR
from src.data.preprocessing import preprocess_pipeline
from src.data.transforms import get_transforms


class DiabeticRetinopathyDataset(Dataset):
    """
    Diabetic retinopathy dataset.

    Args:
        img_dir (str | Path): Directory containing all raw retinal images.
        annotations_df (pd.DataFrame): DataFrame containing image annotations.
            Must include 'id_code' (image file identifiers without extension)
            and 'diagnosis' (target class labels).
        preprocessed_dir (str | Path | None, optional): Directory containing preprocessed
            and cached PNG images. If provided, loads from cache. Otherwise, preprocessing
            is performed on the fly. Defaults to None.
        transform (Callable | None, optional): Transform to apply to the images.
            If None, defaults to `get_transforms(split=split)`. Defaults to None.
        split (str, optional): Dataset split ('train', 'val', or 'test').
            Defaults to 'train'.
    """
    def __init__(self, img_dir: str | Path,
                 annotations_df: pd.DataFrame,
                 preprocessed_dir: str | Path | None = None,
                 transform: Callable | None = None,
                 split: str = "train") -> None:

        self.img_dir = Path(img_dir)
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None
        self.transform = (
            transform if transform is not None else get_transforms(split=split)
        )
        self.split = split
        self.image_ids = annotations_df["id_code"].values
        self.targets = annotations_df["diagnosis"].values

        self.cache_misses = 0

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_id = self.image_ids[idx]
        img = None

        if self.preprocessed_dir is not None:
            cache_path = self.preprocessed_dir / f"{img_id}.png"
            if cache_path.exists():
                img_bgr = cv2.imread(str(cache_path))
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                self.cache_misses += 1
                if self.cache_misses == 1:
                    warnings.warn(
                        f"Cache miss detected. Preprocessed directory is set, but image '{img_id}' "
                        f"was not found in cache. Falling back to on-the-fly preprocessing.",
                        UserWarning
                    )

        if img is None:
            img_path = self.img_dir / f"{img_id}.png"
            img = preprocess_pipeline(img_path)

        label = torch.tensor(self.targets[idx], dtype=torch.long)
        img = self.transform(img)
        return {'image': img, 'label': label}


def get_dataloader(dataset: DiabeticRetinopathyDataset,
                   batch_size : int = 32,
                   num_workers: int = 2) -> DataLoader:
    """
    Returns a PyTorch DataLoader for the given dataset.

    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 2.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(dataset.split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


if __name__ == "__main__":
    train_df = pd.read_csv(SPLITS_DIR / "train_split.csv")
    train_dataset = DiabeticRetinopathyDataset(
        img_dir=str(IMG_DIR),
        annotations_df=train_df,
        split="train",
        preprocessed_dir=PROCESSED_DIR / "images_processed"
    )
    train_loader = get_dataloader(train_dataset, batch_size=8, num_workers=0)

    batch = next(iter(train_loader))
    images = batch["image"]
    labels = batch["label"]

    print(f"Batch images shape: {images.shape}")  # [8, 3, 384, 384]
    print(f"Batch images dtype: {images.dtype}")  # torch.float32
    print(f"Batch labels shape: {labels.shape}")  # [8]
    print(f"Batch labels dtype: {labels.dtype}")  # torch.int64
    print(f"Labels: {labels}")