import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import os

MODEL_PATH = "models/best_xgb.pkl"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("No trained model found, run train_model first.")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="XGBoost Prediction API", version="1.0")


class PredictionInput(BaseModel):
    """Request schema for inference."""
    gas1: float
    datetime: str
    other_feature1: float
    other_feature2: float


@app.get("/")
def root():
    return {"message": "XGBoost Prediction API."}


@app.post("/predict/")
def predict(input_data: PredictionInput):
    """Perform inference using trained model."""
    df = pd.DataFrame([input_data.model_dump()])
    prediction = model.predict(df)
    return {"prediction": float(prediction[0])}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
