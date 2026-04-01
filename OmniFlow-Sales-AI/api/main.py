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
from src.training import MODEL_PATH, REFERENCE_STATS_PATH, get_champion_metrics, train_model, load_champion_model
from src.drift import detect_drift
from src.retrain import retrain_model

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}
TRAINING_JOBS_LOCK = threading.Lock()
DRIFT_JOBS: Dict[str, Dict[str, Any]] = {}
DRIFT_JOBS_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _update_job(job_id: str, **updates) -> None:
    with TRAINING_JOBS_LOCK:
        if job_id in TRAINING_JOBS:
            TRAINING_JOBS[job_id].update(updates)


def _update_drift_job(job_id: str, **updates) -> None:
    with DRIFT_JOBS_LOCK:
        if job_id in DRIFT_JOBS:
            DRIFT_JOBS[job_id].update(_json_safe(updates))


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas objects into plain JSON-serializable Python types."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


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
# POST /drift-retrain (+ async /drift/start)
# ──────────────────────────────────────────────────────────────────────────────

def _execute_drift_retrain(
    contents: bytes,
    target_col: str,
    progress_callback=None,
) -> Dict[str, Any]:
    if not REFERENCE_STATS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No reference stats found. Train an initial model first.",
        )

    if progress_callback:
        progress_callback(10, "preprocessing", "Preparing and validating incoming data...")

    try:
        merged_new = build_training_frame(train_source=contents, target_col=target_col)
        X_new = merged_new.drop(columns=[target_col, "date"], errors="ignore")
        logger.info("New data preprocessed successfully for drift detection")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Preprocessing failed")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

    if progress_callback:
        progress_callback(30, "drift", "Analyzing feature drift against reference baseline...")

    try:
        drift_detected, drift_report = detect_drift(X_new)
    except Exception as e:
        logger.exception("Drift detection failed")
        raise HTTPException(status_code=500, detail=f"Drift detection error: {str(e)}")

    if progress_callback:
        progress_callback(55, "metrics", "Scoring champion model on the uploaded data...")

    champion_metrics_baseline = get_champion_metrics()
    new_data_metrics = None
    performance_degraded = False

    if target_col in merged_new.columns:
        try:
            y_new = merged_new[target_col].astype("float64")
            champion_model = load_champion_model()
            y_pred = champion_model.predict(X_new)

            r2_new = r2_score(y_new, y_pred)
            mae_new = mean_absolute_error(y_new, y_pred)
            rmse_new = np.sqrt(mean_squared_error(y_new, y_pred))

            new_data_metrics = {
                "r2": round(float(r2_new), 4),
                "mae": round(float(mae_new), 4),
                "rmse": round(float(rmse_new), 4),
            }

            baseline_r2 = champion_metrics_baseline.get("r2", 0)
            baseline_mae = champion_metrics_baseline.get("mae", float("inf"))
            baseline_rmse = champion_metrics_baseline.get("rmse", float("inf"))

            # Calculate percentage changes
            r2_change_pct = ((r2_new - baseline_r2) / abs(baseline_r2) * 100) if baseline_r2 != 0 else 0
            mae_change_pct = ((mae_new - baseline_mae) / baseline_mae * 100) if baseline_mae > 0 else 0
            rmse_change_pct = ((rmse_new - baseline_rmse) / baseline_rmse * 100) if baseline_rmse > 0 else 0

            # Absolute drop for guardrail comparison
            r2_drop = baseline_r2 - r2_new
            mae_increase = mae_change_pct
            rmse_increase = rmse_change_pct

            performance_degraded = (r2_drop > 0.05) or (mae_increase > 10) or (rmse_increase > 10)

            logger.info(
                "Champion on new data: R2=%.4f, MAE=%.4f, RMSE=%.4f | "
                "R2 change=%.2f%%, MAE change=%.2f%%, RMSE change=%.2f%%",
                r2_new,
                mae_new,
                rmse_new,
                r2_change_pct,
                mae_change_pct,
                rmse_change_pct,
            )

            # Add percentage changes to metrics
            new_data_metrics["r2_change_pct"] = round(float(r2_change_pct), 2)
            new_data_metrics["mae_change_pct"] = round(float(mae_change_pct), 2)
            new_data_metrics["rmse_change_pct"] = round(float(rmse_change_pct), 2)
        except Exception as e:
            logger.warning("Could not calculate champion metrics on new data: %s", e)

    if progress_callback:
        progress_callback(75, "decision", "Evaluating self-healing strategy...")

    response: Dict[str, Any] = {
        "drift_detected": bool(drift_detected),
        "drift_ratio": float(drift_report["drift_ratio"]),
        "drift_summary": {
            "drifted_features": drift_report["drifted_features"],
            "drift_ratio": float(drift_report["drift_ratio"]),
            "feature_report": _json_safe(drift_report["feature_report"]),
        },
        "champion_metrics_baseline": champion_metrics_baseline,
        "new_data_metrics": new_data_metrics,
        "performance_degraded": bool(performance_degraded),
        "retrain_triggered": False,
        "retrain_result": None,
        "recommendation": "Champion Retained",
        "decision_reason": "No retraining condition was met.",
    }

    should_retrain = (
        (drift_detected and performance_degraded)
        or (drift_report["drift_ratio"] > 0.75)
    )

    if should_retrain:
        if progress_callback:
            progress_callback(88, "retraining", "Training challenger model for self-healing...")
        try:
            retrain_result = retrain_model(contents, target_col=target_col)
            response["retrain_triggered"] = True
            response["retrain_result"] = retrain_result

            if retrain_result.get("replaced"):
                response["recommendation"] = "Challenger Deployed (Self-Healed)"
                response["decision_reason"] = (
                    "Drift was high and challenger outperformed champion by the promotion threshold."
                )
            else:
                response["recommendation"] = "Champion Retained"
                response["decision_reason"] = (
                    "Drift triggered retraining, but challenger did not beat champion by the promotion threshold."
                )
        except Exception as e:
            logger.exception("Retraining failed")
            response["retrain_error"] = str(e)
            response["recommendation"] = "Champion Retained (Retrain Failed)"
            response["decision_reason"] = "Retraining failed due to an internal error; champion kept for safety."
    else:
        if drift_detected and not performance_degraded:
            response["decision_reason"] = "Drift detected, but champion metrics stayed within guardrails."
        elif not drift_detected:
            response["decision_reason"] = "No significant drift detected."
        else:
            response["decision_reason"] = "Drift detected with mild impact; retraining threshold not reached."

    if progress_callback:
        progress_callback(100, "done", "Drift analysis completed.")

    return _json_safe(response)


def _run_drift_job(job_id: str, contents: bytes, target_col: str) -> None:
    def progress_callback(progress: int, stage: str, message: str) -> None:
        _update_drift_job(job_id, progress=progress, stage=stage, message=message, updated_at=_now_iso())

    try:
        _update_drift_job(job_id, status="running", started_at=_now_iso())
        result = _execute_drift_retrain(contents, target_col, progress_callback=progress_callback)
        _update_drift_job(
            job_id,
            status="completed",
            progress=100,
            stage="done",
            message="Drift analysis completed successfully.",
            result=result,
            completed_at=_now_iso(),
            updated_at=_now_iso(),
        )
    except HTTPException as e:
        _update_drift_job(
            job_id,
            status="failed",
            stage="failed",
            message=e.detail,
            error=e.detail,
            completed_at=_now_iso(),
            updated_at=_now_iso(),
        )
    except Exception as e:
        logger.exception("Async drift-retrain failed for job %s", job_id)
        _update_drift_job(
            job_id,
            status="failed",
            stage="failed",
            message=str(e),
            error=f"Drift-retrain error: {str(e)}",
            completed_at=_now_iso(),
            updated_at=_now_iso(),
        )


@app.post("/drift/start", tags=["Drift & Retraining"])
async def start_drift(
    file: UploadFile = File(..., description="New CSV data to check for drift"),
    target_col: str = Form(default=TARGET_DEFAULT, description="Target column name"),
):
    contents = await file.read()
    job_id = str(uuid.uuid4())

    with DRIFT_JOBS_LOCK:
        DRIFT_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "stage": "queued",
            "message": "Drift analysis request accepted.",
            "target_column": target_col,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_run_drift_job,
        args=(job_id, contents, target_col),
        daemon=True,
    )
    thread.start()

    return {
        "status": "accepted",
        "job_id": job_id,
        "poll_url": f"/drift/status/{job_id}",
    }


@app.get("/drift/status/{job_id}", tags=["Drift & Retraining"])
def drift_status(job_id: str):
    with DRIFT_JOBS_LOCK:
        job = DRIFT_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Drift job not found")
    return _json_safe(job)

@app.post("/drift-retrain", tags=["Drift & Retraining"])
async def drift_retrain(
    file: UploadFile = File(..., description="New CSV data to check for drift"),
    target_col: str = Form(default=TARGET_DEFAULT, description="Target column name"),
):
    """
    Synchronous drift+retrain endpoint for backward compatibility.
    Prefer /drift/start + /drift/status/{job_id} for progress-aware UX.
    """
    contents = await file.read()
    return _execute_drift_retrain(contents, target_col)
