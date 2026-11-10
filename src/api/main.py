from fastapi import FastAPI

from src.inference import InferencePipeline, REG_MODEL_PATH, CLF_MODEL_PATH, DEVICE

# Initialize FastAPI app
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

from src.api.routes import router
app.include_router(router)