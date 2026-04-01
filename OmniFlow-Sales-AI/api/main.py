"""Main FastAPI application for OmniFlow Sales AI backend."""

import logging
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

# Ensure project root is on path so `src` is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional

import pandas as pd

from src.preprocessing import TARGET_DEFAULT, build_training_frame
from src.inference import predict_from_payload
from src.training import MODEL_PATH, REFERENCE_STATS_PATH, get_champion_metrics, train_model
from src.drift import detect_drift
from src.retrain import retrain_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}
TRAINING_JOBS_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _update_job(job_id: str, **updates) -> None:
    with TRAINING_JOBS_LOCK:
        if job_id in TRAINING_JOBS:
            TRAINING_JOBS[job_id].update(updates)


def _run_training_job(job_id: str, train_source: Optional[bytes], target_col: str) -> None:
    def progress_callback(progress: int, stage: str, message: str) -> None:
        _update_job(job_id, progress=progress, stage=stage, message=message, updated_at=_now_iso())

    try:
        _update_job(job_id, status="running", started_at=_now_iso())
        metrics = train_model(
            train_source=train_source,
            target_col=target_col,
            progress_callback=progress_callback,
        )
        result = {
            "status": "success",
            "message": "Champion model trained and saved successfully.",
            "target_column": target_col,
            "metrics": metrics,
        }
        _update_job(
            job_id,
            status="completed",
            progress=100,
            stage="done",
            message="Training completed successfully.",
            result=result,
            completed_at=_now_iso(),
            updated_at=_now_iso(),
        )
    except Exception as e:
        logger.exception("Async training failed for job %s", job_id)
        _update_job(
            job_id,
            status="failed",
            stage="failed",
            message=str(e),
            error=f"Training error: {str(e)}",
            completed_at=_now_iso(),
            updated_at=_now_iso(),
        )

# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OmniFlow Sales AI",
    description="Store Sales MLOps API: Training, Prediction, Drift & Retraining",
    version="2.0.0",
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
    return {"status": "ok", "message": "OmniFlow Store Sales API is running"}


@app.get("/health", tags=["Health"])
def health():
    model_ready = MODEL_PATH.exists()
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
    file: Optional[UploadFile] = File(default=None, description="Optional train.csv upload"),
    target_col: str = Form(default=TARGET_DEFAULT, description="Target column name"),
):
    """
    Train champion model for Store Sales.

    If `file` is not provided, backend loads `data/raw/train.csv`.
    Auxiliary sources (`stores.csv`, `oil.csv`, `holidays_events.csv`) are loaded from `data/raw/`.
    """
    try:
        train_source = await file.read() if file else None
        metrics = train_model(train_source=train_source, target_col=target_col)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

    return {
        "status": "success",
        "message": "Champion model trained and saved successfully.",
        "target_column": target_col,
        "metrics": metrics,
    }


@app.post("/train/start", tags=["Training"])
async def start_train(
    file: Optional[UploadFile] = File(default=None, description="Optional train.csv upload"),
    target_col: str = Form(default=TARGET_DEFAULT, description="Target column name"),
):
    train_source = await file.read() if file else None
    job_id = str(uuid.uuid4())

    with TRAINING_JOBS_LOCK:
        TRAINING_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "stage": "queued",
            "message": "Training request accepted.",
            "target_column": target_col,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_run_training_job,
        args=(job_id, train_source, target_col),
        daemon=True,
    )
    thread.start()

    return {
        "status": "accepted",
        "job_id": job_id,
        "poll_url": f"/train/status/{job_id}",
    }


@app.get("/train/status/{job_id}", tags=["Training"])
def train_status(job_id: str):
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


# ──────────────────────────────────────────────────────────────────────────────
# POST /predict
# ──────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    date: str
    store_nbr: int
    family: str
    onpromotion: float = 0
    city: str = "Unknown"
    state: str = "Unknown"
    type: str = "Unknown"
    cluster: int = 0
    dcoilwtico: float = 0.0
    holiday_type: str = "None"
    lag_7: Optional[float] = None


class PredictEnvelope(BaseModel):
    features: Dict[str, Any]


@app.post("/predict", tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Lightweight prediction endpoint without raw CSV merging.

    Send either a direct feature object or use `/predict-legacy` envelope style.
    """
    try:
        prediction = predict_from_payload(request.model_dump())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No champion model found. Train first.")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return {
        "status": "success",
        "prediction": round(prediction, 4),
        "model": "champion",
    }


@app.post("/predict-legacy", tags=["Prediction"])
def predict_legacy(request: PredictEnvelope):
    """Backward-compatible wrapper for old frontend payload format."""
    try:
        prediction = predict_from_payload(request.features)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No champion model found. Train first.")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    return {"status": "success", "prediction": round(prediction, 4), "model": "champion"}


# ──────────────────────────────────────────────────────────────────────────────
# POST /drift-retrain
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/drift-retrain", tags=["Drift & Retraining"])
async def drift_retrain(
    file: UploadFile = File(..., description="New CSV data to check for drift"),
    target_col: str = Form(default=TARGET_DEFAULT, description="Target column name"),
):
    """
    Upload new production data. The system will:
    1. Preprocess data (apply same encoding as training).
    2. Detect statistical drift vs. training baseline.
    3. If drift is detected → train challenger model.
    4. Compare challenger vs champion → promote if better.
    """
    contents = await file.read()

    if not REFERENCE_STATS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No reference stats found. Train an initial model first.",
        )

    try:
        merged_new = build_training_frame(train_source=contents, target_col=target_col)
        X_new = merged_new.drop(columns=[target_col, "date"], errors="ignore")
        logger.info("New data preprocessed successfully for drift detection")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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
            retrain_result = retrain_model(contents, target_col=target_col)
            response["retrain_triggered"] = True
            response["retrain_result"] = retrain_result
        except Exception as e:
            logger.exception("Retraining failed")
            response["retrain_error"] = str(e)

    return response
