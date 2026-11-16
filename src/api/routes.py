import asyncio
import base64
import contextlib
import io
import logging
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
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
        request (Request): The FastAPI request object.

    Returns:
        InferencePipeline: The inference pipeline
            instance stored in the application state.
    """
    return request.app.state.pipeline


def encode_array(arr: np.ndarray, to_uint8: bool = False) -> dict:
    """
    Encodes a NumPy array into a base64 string.

    Args:
        arr (np.ndarray): The array to encode.
        to_uint8 (bool): Whether to convert floating arrays to uint8.

    Returns:
        dict: A dictionary containing:
            - "data" (str): The base64-encoded array data.
            - "dtype" (str): The data type of the array.
            - "shape" (list): The shape of the array.
    """
    if to_uint8 and np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)

    contiguous_arr = np.ascontiguousarray(arr)
    return {
        "data": base64.b64encode(contiguous_arr).decode("utf-8"),
        "dtype": str(contiguous_arr.dtype),
        "shape": list(contiguous_arr.shape)
    }


@router.get("/")
def root() -> dict:
    return {"message": "Welcome to the Diabetic Retinopathy API. Use /docs to see interactive documentation."}


@router.get("/health")
def health_check(request: Request) -> dict:
    """
    Checks the health status of the API and if the model pipeline is loaded.

    Args:
        request (Request): The FastAPI request object.

    Returns:
        dict: A dictionary with API health status and model loading state.

    Raises:
        HTTPException: If the model pipeline is not loaded.
    """
    # Check whether pipeline is initialized in the app
    if not hasattr(request.app.state, "pipeline") or request.app.state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model pipeline is not loaded yet."
        )

    return {
        "status": "healthy",
        "model_loaded": True
    }


@contextlib.asynccontextmanager
async def validate_and_save_upload(file: UploadFile) -> Path:
    """
    Validates the uploaded file, checks if it is a valid image,
    and saves it temporarily for processing.

    Args:
        file (UploadFile): The uploaded image file.

    Yields:
        Path: The path to the temporarily saved image file.

    Raises:
        HTTPException: If the file is too large or not a valid image.
    """
    # Read up to MAX_FILE_SIZE + 1 bytes and check size
    contents = await file.read(MAX_FILE_SIZE + 1)
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File is too large. Maximum allowed size is 10 MB."
        )

    # Verify given file is an image
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. The file is corrupted or not a supported image format."
        ) from e

    # Temporary file to pass its path to inference method
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(contents)
        temp_path = Path(temp_file.name)

    try:
        # Yield temporary file path
        yield temp_path
    finally:
        # Delete temporary file
        if temp_path.exists():
            temp_path.unlink()


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...),
                        pipeline: InferencePipeline = Depends(get_pipeline)) -> dict:
    """
    Returns a prediction for diabetic retinopathy.

    Args:
        file (UploadFile): The uploaded image file.
        pipeline (InferencePipeline): The inference pipeline dependency.

    Returns:
        dict: A dictionary containing:
            - "filename" (str): The name of the uploaded file.
            - "prediction" (dict): The prediction for the uploaded file.

    Raises:
        HTTPException: If the file is too large, invalid, or an internal error occurs.
    """
    try:
        async with validate_and_save_upload(file) as temp_path:
            # Separate thread to avoid blocking main event loop
            prediction_result = await asyncio.to_thread(
                pipeline.predict_single, temp_path
            )

        return {
            "filename": file.filename,
            "prediction": prediction_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        # Log all unexpected errors on the server, while
        # returning 500 error to the client without details
        logger.exception("An unexpected error occurred during prediction processing.")
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred while processing the image."
        ) from e