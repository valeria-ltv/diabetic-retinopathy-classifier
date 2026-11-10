from pydantic import BaseModel, Field


class PredictionDetails(BaseModel):
    regression_pred: float = Field(..., description="Raw regression model prediction.")
    classification_probs: list[float] = Field(..., description="Softmax probabilities for each class.")
    class_score: float = Field(..., description="Expected value derived from classification probabilities.")
    ensemble_score: float = Field(..., description="Weighted linear combination of class score and regression prediction.")
    predicted_class: int = Field(..., description="Final discretized predicted class.")


class PredictionResponse(BaseModel):
    filename: str = Field(..., description="Name of the uploaded image file.")
    prediction: PredictionDetails