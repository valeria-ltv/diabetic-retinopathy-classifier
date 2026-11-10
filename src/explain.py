from pathlib import Path

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, RawScoresOutputTarget

from src.inference import InferencePipeline
from src.data.preprocessing import preprocess_pipeline


def generate_explanations(img_path: str | Path,
                          inference_pipeline: InferencePipeline,
                          use_smoothing: bool = True) -> dict:
    """
    Generate Grad-CAM explanations for an input image
    using both classifier and regressor models.

    Args:
        img_path (str | Path): Path to the image used for inference.
        inference_pipeline (src.inference.InferencePipeline): Initialized inference pipeline
            containing both models and ensemble weights.
        use_smoothing (bool, optional): WheWhether to apply eigen-smoothing to remove noise
            from the Grad-CAM heatmaps. Defaults to True.

    Returns:
        dict: A dictionary containing:
            - "preprocessed" (np.ndarray): Preprocessed input image normalized to float [0, 1].
            - "cam_clf" (np.ndarray): Grad-CAM heatmap overlay for the classifier's predicted class.
            - "cam_reg" (np.ndarray): Grad-CAM heatmap overlay for the regressor model.
    """
    result = inference_pipeline.predict_single(img_path)
    clf_probs = torch.tensor(result["classification_probs"])
    clf_pred = torch.argmax(clf_probs).item()

    # Preprocessing
    img_np = preprocess_pipeline(img_path)
    img_float = img_np.astype(np.float32) / 255.0
    img_tensor = inference_pipeline.transform(img_np).unsqueeze(0).to(inference_pipeline.device)

    # Generate Grad-CAM for classifier (predicted class)
    target_layers_clf = [inference_pipeline.model_clf.model.conv_head]
    targets_clf = [ClassifierOutputTarget(int(clf_pred))]
    with GradCAM(model=inference_pipeline.model_clf, target_layers=target_layers_clf) as cam:
        cam_clf = cam(
            input_tensor=img_tensor,
            targets=targets_clf,
            eigen_smooth=use_smoothing,
        )[0]
        vis_clf = show_cam_on_image(img_float, cam_clf, use_rgb=True)

    # Generate Grad-CAM for regressor
    target_layers_reg = [inference_pipeline.model_reg.model.conv_head]
    targets_reg = [RawScoresOutputTarget()]
    with GradCAM(model=inference_pipeline.model_reg, target_layers=target_layers_reg) as cam:
        cam_reg = cam(
            input_tensor=img_tensor,
            targets=targets_reg,
            eigen_smooth=use_smoothing,
        )[0]
        vis_reg = show_cam_on_image(img_float, cam_reg, use_rgb=True)

    return {
        "preprocessed": img_float,
        "cam_clf": vis_clf,
        "cam_reg": vis_reg,
    }