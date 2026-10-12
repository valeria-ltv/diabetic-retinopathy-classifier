# Data

## Source
APTOS 2019 Blindness Detection dataset from Kaggle:
https://www.kaggle.com/competitions/aptos2019-blindness-detection

## Description
A clinician has rated each image for the severity of diabetic retinopathy on a scale of 0 to 4:
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

## How to download

1. Install Kaggle API and configure credentials (kaggle.json in ~/.kaggle/)
2. Join the competition on Kaggle (accept rules) if not already joined
3. Run:

```bash
pip install kaggle
cd data/raw
kaggle competitions download -c aptos2019-blindness-detection
unzip aptos2019-blindness-detection.zip -d aptos2019
```

## Structure after download

```
data/raw/aptos2019/
├── train_images/
├── test_images/
├── train.csv
├── test.csv
└── sample_submission.csv
```

## Note
Raw and processed data are excluded from version control (see .gitignore).
This README documents how to reproduce the data setup locally.

Train/val/test split CSVs (data/processed/aptos2019/splits/) are tracked in git 
to ensure reproducibility of the exact split across environments

## Processed data structure

```
data/processed/aptos2019/
├── images_processed/      # cached preprocessed images (not tracked in git)
└── splits/
    ├── train_split.csv    # id_code + diagnosis, ~70% stratified
    ├── val_split.csv      # ~15% stratified
    └── test_split.csv     # ~15% stratified
```

## Preprocessed image cache

The preprocessing pipeline is deterministic and expensive: loading the raw 
images through it on the fly took ~2.3 s per batch of 32, which would leave 
the GPU idle during training. Instead, the pipeline is run once per image 
and the results are cached as PNGs in `images_processed/`.

To generate the cache (from the repository root):

```
python -m src.data.cache_preprocessed
```

Augmentation is not cached: it is applied on the fly after reading from the cache.
At inference time (deployment) the pipeline runs on the fly on a single raw image.