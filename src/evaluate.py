"""
Evaluation script for Diabetic Retinopathy classification & regression ensemble.

This script performs the following steps:
    1. Loads the test dataset split from `test_split.csv`.
    2. Initializes the inference pipeline using pre-trained regression
       and classification model checkpoints.
    3. Iterates through each test image and runs inference on it.
    4. Computes and prints the Quadratic Weighted Kappa (QWK) score.
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from src.config import IMG_DIR, SPLITS_DIR
from src.inference import InferencePipeline, REG_MODEL_PATH, CLF_MODEL_PATH, DEVICE

if __name__ == "__main__":
    test_df = pd.read_csv(SPLITS_DIR / "test_split.csv")
    inference_pipeline = InferencePipeline(reg_model_path=REG_MODEL_PATH,
                                           clf_model_path=CLF_MODEL_PATH,
                                           device=DEVICE)

    targets = []
    preds = []

    for row in tqdm(test_df.itertuples(), total=len(test_df), desc="Inference"):
        img_path = IMG_DIR / f"{getattr(row, 'id_code')}.png"
        result = inference_pipeline.predict_single(img_path)
        targets.append(getattr(row, "diagnosis"))
        preds.append(result["predicted_class"])

    qwk = cohen_kappa_score(targets, preds, weights='quadratic')
    print(f"Test QWK: {qwk:.4f}")