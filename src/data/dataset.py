import os
from collections.abc import Callable

import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing import preprocess_pipeline
from transforms import get_transforms


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

if __name__ == "__main__":
    df = pd.read_csv("../../data/processed/aptos2019/splits/train_split.csv")
    dataset = DiabeticRetinopathyDataset(img_dir="../../data/raw/aptos2019/train_images", annotations_df=df)
    sample = next(iter(dataset))
    print(sample["image"].shape)
    print(sample["image"].dtype)
    print(sample["label"].dtype)
    print(sample["label"])