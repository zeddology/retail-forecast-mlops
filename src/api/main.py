import sys
from pathlib import Path

HERE = Path(__file__).parent          # src/api
sys.path.insert(0, str(HERE.parent))  # src/  -> makes `config` importable
sys.path.insert(0, str(HERE))         # src/api/ -> makes `schemas` importable

import mlflow
import mlflow.lightgbm
import pandas as pd
from fastapi import FastAPI
from schemas import ForecastRequest, ForecastResponse
from config import config

# ── Create the FastAPI app ──────────────────────────────────────
app = FastAPI(title="Retail Demand Forecast API")

# ── Load the model ONCE at startup ──────────────────────────────
# (loaded into memory when the server starts, reused for every request)
MODEL = None

@app.on_event("startup")
def load_model():
    global MODEL
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    # Load the latest model from the experiment's most recent run
    experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    run_id = runs.iloc[0]["run_id"]
    MODEL = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
    print(f"Loaded model from run {run_id}")

# ── Health check endpoint ───────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

# ── Forecast endpoint ───────────────────────────────────────────
@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    # Convert the incoming features into a one-row DataFrame
    # (the model was trained on a DataFrame, so column order must match)
    features = pd.DataFrame([request.dict()])
    prediction = MODEL.predict(features)[0]
    return ForecastResponse(predicted_sales=round(float(prediction), 2))