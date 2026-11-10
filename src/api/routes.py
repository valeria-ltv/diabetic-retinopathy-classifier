import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException

from src.api.main import pipeline
from src.api.schemas import PredictionResponse

router = APIRouter()


@router.get("/")
def root():
    return {"message": "Welcome to the Diabetic Retinopathy API. Use /docs to see interactive documentation."}


@router.post("/predict", response_model=PredictionResponse)
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