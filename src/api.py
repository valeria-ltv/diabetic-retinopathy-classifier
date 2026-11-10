import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field

from src.inference import InferencePipeline, REG_MODEL_PATH, CLF_MODEL_PATH, DEVICE

# Initialize FastAPI application
app = FastAPI(
    title="Diabetic Retinopathy Diagnosis & XAI API",
    description="API for inference and Grad-CAM explanations of hybrid DR models.",
    version="1.0"
)

# Load inference pipeline once when server starts
# to avoid loading models for every request
print("Loading models into memory...")
pipeline = InferencePipeline(
    reg_model_path=REG_MODEL_PATH,
    clf_model_path=CLF_MODEL_PATH,
    device=DEVICE
)
print("Models loaded successfully.")

# Pydantic schemas
class PredictionDetails(BaseModel):
    regression_pred: float = Field(..., description="Raw regression model prediction.")
    classification_probs: list[float] = Field(..., description="Softmax probabilities for each class.")
    class_score: float = Field(..., description="Expected value derived from classification probabilities.")
    ensemble_score: float = Field(..., description="Weighted linear combination of class score and regression prediction.")
    predicted_class: int = Field(..., description="Final discretized predicted class.")


class PredictionResponse(BaseModel):
    filename: str = Field(..., description="Name of the uploaded image file.")
    prediction: PredictionDetails


@app.get("/")
def root():
    return {"message": "Welcome to the Diabetic Retinopathy API. Use /docs to see interactive documentation."}


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    An endpoint for obtaining predictions for an uploaded retinal image.
    """
    # Check if given file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read bytes of the downloaded file
        contents = await file.read()

        # Temporary file to pass its path to inference method
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(contents)
            temp_path = Path(temp_file.name)

        try:
            # Inference
            prediction_result = pipeline.predict_single(temp_path)

        finally:
            # Delete temporary file
            if temp_path.exists():
                temp_path.unlink()

        return {
            "filename": file.filename,
            "prediction": prediction_result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e