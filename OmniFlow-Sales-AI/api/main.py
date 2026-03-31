"""Main FastAPI application for OmniFlow Sales AI MLOps Backend."""

import logging
import sys
from pathlib import Path

# Ensure project root is on path so `src` is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional
import json

from src.ingestion import load_raw_data
from src.preprocessing import preprocess_data, split_data, REFERENCE_STATS_PATH, FEATURE_COLS_PATH
from src.training import train_model, load_champion_model, get_champion_metrics
from src.drift import detect_drift
from src.retrain import retrain_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OmniFlow Sales AI",
    description="MLOps API: Train, Predict, Drift Detection & Automated Retraining",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "OmniFlow Sales AI API is running 🚀"}


@app.get("/health", tags=["Health"])
def health():
    model_ready = (PROJECT_ROOT / "models" / "champion_model.pkl").exists()
    stats_ready = REFERENCE_STATS_PATH.exists()
    return {
        "api": "healthy",
        "model_ready": model_ready,
        "reference_stats_ready": stats_ready,
        "champion_metrics": get_champion_metrics() if model_ready else {},
    }


# ──────────────────────────────────────────────────────────────────────────────
# POST /train
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/train", tags=["Training"])
async def train(
    file: UploadFile = File(..., description="CSV training data"),
    target_col: str = Form(..., description="Name of the target column"),
):
    """
    Train a new champion model on uploaded CSV data.

    - Preprocesses data (imputation + encoding)
    - Trains RandomForestRegressor
    - Saves model, encoders, feature list, reference stats
    - Logs experiment to MLflow
    """
    try:
        contents = await file.read()
        df = load_raw_data(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' not found. Available: {df.columns.tolist()}",
        )

    try:
        X, y = preprocess_data(df, target_col=target_col, is_training=True)
        X_train, X_test, y_train, y_test = split_data(X, y)
        metrics = train_model(X_train, X_test, y_train, y_test, model_name="champion")
    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

    return {
        "status": "success",
        "message": "Model trained and saved successfully.",
        "target_column": target_col,
        "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "metrics": metrics,
    }


# ──────────────────────────────────────────────────────────────────────────────
# POST /predict
# ──────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: Dict[str, Any]


@app.post("/predict", tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Generate a prediction from the champion model.

    Send a JSON body: `{ "features": { "col1": val1, "col2": val2, ... } }`
    """
    try:
        model = load_champion_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Load expected feature columns
    if not FEATURE_COLS_PATH.exists():
        raise HTTPException(status_code=500, detail="Feature column list not found. Retrain the model.")

    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)

    import pandas as pd
    row = pd.DataFrame([request.features])

    # Preprocess (inference mode — no refitting)
    try:
        X, _ = preprocess_data(row, target_col="__no_target__", is_training=False)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {str(e)}")

    # Align with training feature columns
    X = X.reindex(columns=feature_cols, fill_value=0)

    prediction = model.predict(X)[0]

    return {
        "status": "success",
        "prediction": round(float(prediction), 4),
        "model": "champion",
    }


# ──────────────────────────────────────────────────────────────────────────────
# POST /drift-retrain
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/drift-retrain", tags=["Drift & Retraining"])
async def drift_retrain(
    file: UploadFile = File(..., description="New CSV data to check for drift"),
    target_col: str = Form(..., description="Target column name"),
):
    """
    Upload new production data. The system will:
    1. Preprocess data (apply same encoding as training).
    2. Detect statistical drift vs. training baseline.
    3. If drift is detected → train challenger model.
    4. Compare challenger vs champion → promote if better.
    """
    try:
        contents = await file.read()
        new_df = load_raw_data(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not REFERENCE_STATS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No reference stats found. Train an initial model first.",
        )

    # ── Step 1: Preprocess new data for drift detection ──────────────────────
    # Apply the same transformations (encoding, etc.) that were applied during training
    try:
        X_new, _ = preprocess_data(new_df, target_col=target_col, is_training=False)
        logger.info("New data preprocessed successfully for drift detection")
    except Exception as e:
        logger.exception("Preprocessing failed")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

    # ── Step 2: Drift Detection ──────────────────────────────────────────────
    # Now detect drift using the preprocessed features (numeric columns aligned with reference stats)
    try:
        drift_detected, drift_report = detect_drift(X_new)
    except Exception as e:
        logger.exception("Drift detection failed")
        raise HTTPException(status_code=500, detail=f"Drift detection error: {str(e)}")

    response = {
        "drift_detected": drift_detected,
        "drift_summary": {
            "drifted_features": drift_report["drifted_features"],
            "drift_ratio": drift_report["drift_ratio"],
            "feature_report": drift_report["feature_report"],
        },
        "retrain_triggered": False,
        "retrain_result": None,
    }

    # ── Step 3: Retraining (only if drift detected) ──────────────────────────
    if drift_detected:
        logger.info("Drift detected — initiating retraining pipeline...")
        try:
            retrain_result = retrain_model(new_df, target_col=target_col)
            response["retrain_triggered"] = True
            response["retrain_result"] = retrain_result
        except Exception as e:
            logger.exception("Retraining failed")
            response["retrain_error"] = str(e)

    return response
