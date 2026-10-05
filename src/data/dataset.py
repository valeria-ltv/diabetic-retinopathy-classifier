import os
from collections.abc import Callable

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.preprocessing import preprocess_pipeline
from src.data.transforms import get_transforms


class DiabeticRetinopathyDataset(Dataset):
    """
    Diabetic retinopathy dataset.

    Args:
        img_dir (str): Directory containing all raw retinal images.
        annotations_df (pd.DataFrame): DataFrame containing image annotations.
            Must include 'id_code' (image file identifiers without extension)
            and 'diagnosis' (target class labels).
        transform (Callable | None, optional): Transform to apply to the images.
            If None, defaults to `get_transforms(split=split)`. Defaults to None.
        split (str, optional): Dataset split ('train', 'val', or 'test').
            Defaults to "train".
    """
    def __init__(self, img_dir: str, annotations_df: pd.DataFrame,
                 transform: Callable | None = None, split: str = "train") -> None:
        self.img_dir = img_dir
        self.split = split
        self.transform = (
            transform if transform is not None else get_transforms(split=split)
        )
        self.image_ids = annotations_df["id_code"].values
        self.targets = annotations_df["diagnosis"].values

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path = os.path.join(self.img_dir, f"{self.image_ids[idx]}.png")
        img = preprocess_pipeline(img_path)
        label = torch.tensor(self.targets[idx], dtype=torch.long)
        img = self.transform(img)
        return {'image': img, 'label': label}


def get_dataloader(dataset: Dataset, split: str = "train",
                   batch_size : int = 32, num_workers: int = 2) -> DataLoader:
    """
    Returns a PyTorch DataLoader for the given dataset.

    Args:
        dataset (Dataset): The dataset to load.
        split (str, optional): Dataset split ('train', 'val', or 'test'). Defaults to "train".
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 2.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


if __name__ == "__main__":
    df = pd.read_csv("../../data/processed/aptos2019/splits/train_split.csv")
    train_dataset = DiabeticRetinopathyDataset(
        img_dir="../../data/raw/aptos2019/train_images",
        annotations_df=df
    )
    train_loader = get_dataloader(train_dataset, split="train", batch_size=8, num_workers=0)

    batch = next(iter(train_loader))
    images = batch["image"]
    labels = batch["label"]

    print(f"Batch images shape: {images.shape}")  # [8, 3, 384, 384]
    print(f"Batch images dtype: {images.dtype}")  # torch.float32
    print(f"Batch labels shape: {labels.shape}")  # [8]
    print(f"Batch labels dtype: {labels.dtype}")  # torch.int64
    print(f"Labels: {labels}")