"""Model training module — trains, evaluates, and persists the ML model."""

import joblib
import json
import logging
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "champion_model.pkl"
METRICS_PATH = MODELS_DIR / "champion_metrics.json"


def train_model(X_train, X_test, y_train, y_test, model_name: str = "champion") -> dict:
    """
    Train a RandomForestRegressor, evaluate it, and persist it.

    Args:
        X_train, X_test: Feature matrices.
        y_train, y_test: Target arrays.
        model_name: Label used in MLflow run naming.

    Returns:
        dict with r2, mae, rmse scores.
    """
    mlflow.set_experiment("OmniFlow-Sales-Prediction")

    with mlflow.start_run(run_name=model_name):
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "r2": round(float(r2_score(y_test, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        }

        # Log to MLflow
        mlflow.log_params({
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info(f"Model trained | R2={metrics['r2']} | MAE={metrics['mae']} | RMSE={metrics['rmse']}")

    # Persist model and metrics
    save_path = MODELS_DIR / f"{model_name}_model.pkl"
    joblib.dump(model, save_path)

    if model_name == "champion":
        joblib.dump(model, MODEL_PATH)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Champion model saved to {MODEL_PATH}")

    return metrics


def load_champion_model():
    """Load the saved champion model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Please train a model first.")
    return joblib.load(MODEL_PATH)


def get_champion_metrics() -> dict:
    """Load the saved champion metrics."""
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)
