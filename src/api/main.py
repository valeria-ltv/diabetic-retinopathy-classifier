import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.inference import InferencePipeline, REG_MODEL_PATH, CLF_MODEL_PATH, DEVICE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the models into app.state
    logger.info("Loading models into memory...")
    app.state.pipeline = InferencePipeline(
        reg_model_path=REG_MODEL_PATH,
        clf_model_path=CLF_MODEL_PATH,
        device=DEVICE
    )
    # Create global asyncio.Lock to synchronize requests to the model
    app.state.model_lock = asyncio.Lock()
    logger.info("Models loaded successfully.")
    yield
    # Release the resources on shutdown
    if hasattr(app.state, "pipeline"):
        del app.state.pipeline
        del app.state.model_lock
        logger.info("Models unloaded from memory.")

# Initialize FastAPI app
app = FastAPI(
    lifespan=lifespan,
    title="Diabetic Retinopathy Classification API",
    description="Educational prototype for diabetic retinopathy screening. "
                "For research and demonstration purposes only. Not for clinical use or medical diagnosis.",
    version="1.0"
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catches all unhandled exceptions across the application, logs the full traceback,
    and returns a generic 500 Internal Server Error without leaking implementation details.
    """
    logger.exception("Unhandled error occurred while processing %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred while processing the request."}
    )

app.include_router(router)