import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.inference import InferencePipeline, REG_MODEL_PATH, CLF_MODEL_PATH, DEVICE
from src.api.routes import router

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
    logger.info("Models loaded successfully.")
    yield
    # Release the resources on shutdown
    if hasattr(app.state, "pipeline"):
        del app.state.pipeline
        logger.info("Models unloaded from memory.")

# Initialize FastAPI app
app = FastAPI(
    lifespan=lifespan,
    title="Diabetic Retinopathy Diagnosis & XAI API",
    description="API for inference and Grad-CAM explanations of hybrid DR models.",
    version="1.0"
)

app.include_router(router)