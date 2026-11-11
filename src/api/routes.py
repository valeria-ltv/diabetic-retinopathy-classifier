import asyncio
import io
import logging
import tempfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Depends

from src.api.schemas import PredictionResponse
from src.inference import InferencePipeline

router = APIRouter()

# Configure logging to record errors on the server
logger = logging.getLogger(__name__)

# Maximum file size allowed for upload: 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024

def get_pipeline(request: Request) -> InferencePipeline:
    """
    Dependency to retrieve the preloaded InferencePipeline from app.state

    Args:
        request (fastapi.Request): The FastAPI request object.

    Returns:
        src.inference.InferencePipeline: The inference pipeline
            instance stored in the application state.
    """
    return request.app.state.pipeline


@router.get("/")
def root() -> dict:
    return {"message": "Welcome to the Diabetic Retinopathy API. Use /docs to see interactive documentation."}


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...),
                        pipeline: InferencePipeline = Depends(get_pipeline)) -> dict:
    """
    An endpoint for obtaining predictions for an uploaded retinal image.

    Args:
        file (UploadFile): The uploaded image file.
        pipeline (src.inference.InferencePipeline): The inference pipeline dependency.

    Returns:
        dict: A dictionary containing the filename and prediction result.

    Raises:
        HTTPException: If the file is too large, invalid, or an internal error occurs.
    """
    try:
        # Read bytes of the downloaded file and check size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File is too large. Maximum allowed size is 10 MB."
            )

        # Verify given file is an image
        try:
            image = Image.open(io.BytesIO(contents))
            image.verify()
        except (UnidentifiedImageError, Exception) as e:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file. The file is corrupted or not a supported image format."
            ) from e

        # Temporary file to pass its path to inference method
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(contents)
            temp_path = Path(temp_file.name)

        try:
            # Inference with asyncio.to_thread
            # Separate thread to avoid blocking main event loop
            prediction_result = await asyncio.to_thread(
                pipeline.predict_single, temp_path
            )

        finally:
            # Delete temporary file
            if temp_path.exists():
                temp_path.unlink()

        return {
            "filename": file.filename,
            "prediction": prediction_result,
        }

    except HTTPException as he:
        raise he

    except Exception as e:
        # Log all unexpected errors on the server, while
        # returning 500 error to the client without details
        logger.exception("An unexpected error occurred during prediction processing.")
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred while processing the image."
        ) from e