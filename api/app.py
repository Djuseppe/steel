import logging

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from xgboost import XGBRegressor

from api.config import settings
from api.inference import inference
from api.models import PredictionInput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="XGBoost Prediction API", version="1.0")

model: None | XGBRegressor = None


@app.on_event("startup")
def load_model():
    """Load the ML model at startup."""
    global model
    if not settings.model_path.exists():
        logger.error("No fitted model found, please train the model first.")
        raise RuntimeError("No fitted model found, please train the model first.")
    model = joblib.load(settings.model_path)
    logger.info("Model successfully loaded.")


@app.get(path="/", response_model=dict, summary="Root endpoint.")
def root() -> dict:
    """Root endpoint."""
    return {"message": "XGBoost Prediction API."}


@app.post("/predict/", response_model=dict, summary="Perform inference using the fitted XGB model.")
async def predict(input_data: PredictionInput) -> dict:
    """Perform inference using the trained model."""
    global model
    if model is None:
        logger.error("Model was not loaded.")
        raise HTTPException(status_code=503, detail="Mode was not loaded.")
    try:
        prediction = inference(input_data=input_data, model=model)
        if np.isnan(prediction):
            logger.warning("Inference returned np.nan.")
            raise HTTPException(status_code=422, detail="Prediction failed.")
        return {"prediction": float(prediction)}
    except HTTPException as exception:
        raise exception
    except Exception as exception:
        logger.error(f"Prediction error: {exception}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
