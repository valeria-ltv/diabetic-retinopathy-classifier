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