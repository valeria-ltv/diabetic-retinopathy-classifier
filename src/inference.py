from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn

from src.config import SPLITS_DIR, IMG_DIR, MODELS_DIR
from src.data.preprocessing import preprocess_pipeline
from src.data.transforms import get_transforms

REG_MODEL_PATH = MODELS_DIR / "effnetb0_reg.pth"
CLF_MODEL_PATH = MODELS_DIR / "effnetb0_clf.pth"

class ModelWrapper(nn.Module):
    """
    Wrapper for EfficientNet-B0 model.

    Args:
       num_outputs (int): Number of outputs for the model.
    """
    def __init__(self, num_outputs: int):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_outputs)

    def forward(self, x):
        return self.model(x)


class InferencePipeline:
    """
    Inference pipeline.

    Args:
        reg_model_path (str | Path): Path to the regression model weights file.
        clf_model_path (str | Path): Path to the classification model weights file.
        w (float, optional): Weight for combining classification and regression scores.
            Determines the proportion of the classification model inference. Defaults to 0.65.
        thresholds (list[float] | None, optional): List of thresholds for class assignment. Defaults to None.
            Defaults to optimized thresholds on validation set ([0.5310, 1.5119, 2.4416, 3.4799]).
        device (str, optional): Device to run inference on ("cpu" or "cuda"). Defaults to "cpu".
    """
    def __init__(self,
                 reg_model_path: str | Path,
                 clf_model_path: str | Path,
                 w: float = 0.65,
                 thresholds: list[float] | None = None,
                 device: str = "cpu"):
        self.w = w
        self.thresholds = thresholds or [0.5310, 1.5119, 2.4416, 3.4799]
        self.device = torch.device(device)
        self.transform = get_transforms(split="test")

        # Load models
        self.model_reg = self._load_model(num_outputs=1, path=reg_model_path)
        self.model_class = self._load_model(num_outputs=5, path=clf_model_path)

    def _load_model(self, num_outputs: int, path: str | Path) -> nn.Module:
        """
        Loads model from given path.

        Args:
            num_outputs (int): Number of outputs of the model.
            path (str | Path): Path to model weights.
        Returns:
            nn.Module
        """
        net = ModelWrapper(num_outputs=num_outputs)
        net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        net.to(self.device)
        net.eval()
        return net


    def _apply_thresholds(self, scores: np.ndarray, thresholds: list[float]) -> np.ndarray:
        """
        Applies thresholds to scores.

        Args:
            scores (np.ndarray): Scores to apply thresholds to.
            thresholds (list[float]): Thresholds to apply to scores.
        Returns:
            np.ndarray
        """
        return np.digitize(scores, thresholds)

    @torch.no_grad()
    def predict_single(self, img_path: str | Path) -> dict:
        """
        Make prediction for a single image. Processes the image, runs inference with both
        classification and regression models, and combines results using a hybrid ensemble.

        Args:
            img_path (str | Path): Path to the input image.

        Returns:
            dict: Dictionary containing raw regression prediction, classification probabilities,
                class score (expected value derived from classification probabilities),
                ensemble score (linear combination of class score and regression prediction),
                and predicted class.
        """
        # Preprocess an image and apply transforms
        img_np = preprocess_pipeline(img_path)
        tensor = self.transform(img_np).unsqueeze(0).to(self.device)  # [1, 3, H, W]

        # Regression model inference
        reg_pred = self.model_reg(tensor).item()

        # Classification model inference (with softmax for probabilities)
        clf_logits = self.model_class(tensor)
        probs = torch.softmax(clf_logits, dim=1).cpu().numpy()[0]

        # Calculate expected value (soft score) from classification probabilities
        class_values = np.array([0, 1, 2, 3, 4])
        class_score = np.sum(probs * class_values)

        # Hybrid Ensemble
        ensemble_score = self.w * class_score + (1 - self.w) * reg_pred
        final_class = int(self._apply_thresholds(np.array([ensemble_score]), self.thresholds)[0])

        return {
            "regression_pred": reg_pred,
            "classification_probs": probs.tolist(),
            "class_score": class_score,
            "ensemble_score": ensemble_score,
            "predicted_class": final_class
        }


if __name__ == "__main__":
    import random
    import pandas as pd
    import matplotlib.pyplot as plt
    import cv2

    test_df = pd.read_csv(SPLITS_DIR / "test_split.csv")
    random_index = random.randint(0, len(test_df) - 1)

    img_id = test_df.loc[random_index, "id_code"]
    img_label = test_df.loc[random_index, "diagnosis"]
    img_path = IMG_DIR / f"{img_id}.png"

    inference_pipeline = InferencePipeline(reg_model_path=REG_MODEL_PATH,
                                           clf_model_path=CLF_MODEL_PATH)

    prediction = inference_pipeline.predict_single(img_path)

    print(f"Image id: {img_id}")
    print(f"Image label (True): {img_label}")
    print(f"Predicted class: {prediction['predicted_class']}")

    print("\nDetailed scores:")
    for k, v in prediction.items():
        if isinstance(v, list):
            probs = ", ".join([f"{p:.2f}" for p in v])
            print(f"  {k}: [{probs}]")
        else:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"ID: {img_id} | True: {img_label} | Pred: {prediction['predicted_class']}")
    plt.axis("off")
    plt.show()