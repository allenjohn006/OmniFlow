"""Training utilities for Store Sales champion model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_DEFAULT,
    build_training_frame,
    validate_feature_frame,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "champion.joblib"
METRICS_PATH = MODELS_DIR / "champion_metrics.json"
REFERENCE_STATS_PATH = MODELS_DIR / "reference_stats.json"


def _assert_feature_matrix_schema(X: pd.DataFrame, frame_name: str) -> None:
    """Ensure model inputs are exactly the expected feature schema."""
    actual = list(X.columns)
    if actual != FEATURE_COLUMNS:
        raise ValueError(
            f"{frame_name} feature columns do not match expected schema. "
            f"Expected {FEATURE_COLUMNS}, got {actual}"
        )


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ]
    )
    if XGBRegressor is not None:
        model = XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
    else:
        logger.warning("xgboost is unavailable; using HistGradientBoostingRegressor fallback.")
        model = HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _build_reference_stats(df):
    numeric_stats = {
        col: {
            "mean": float(df[col].mean()),
            "std": float(df[col].std() if df[col].std() > 0 else 1e-6),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }
        for col in NUMERICAL_FEATURES
    }
    categorical_stats = {
        col: df[col].astype(str).value_counts(normalize=True).to_dict() for col in CATEGORICAL_FEATURES
    }
    return {"numerical": numeric_stats, "categorical": categorical_stats}


def train_model(
    train_source=None,
    target_col: str = TARGET_DEFAULT,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
) -> dict:
    """
    Train an XGBoost pipeline with a temporal split and persist the champion artifact.

    Args:
        train_source: Optional uploaded bytes/path for train.csv. If None, loads data/raw/train.csv.
        target_col: Target column name.

    Returns:
        dict with r2, mae, rmse scores.
    """
    if progress_callback:
        progress_callback(5, "loading", "Reading data and building training frame...")

    frame = build_training_frame(train_source=train_source, target_col=target_col)

    # Use datetime comparison properly
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", target_col])
    if progress_callback:
        progress_callback(20, "split", "Applying temporal train/test split...")

    train_mask = frame["date"] < pd.Timestamp("2016-01-01")
    train_df = frame[train_mask]
    test_df = frame[~train_mask]
    if train_df.empty or test_df.empty:
        raise ValueError("Temporal split failed. Ensure data includes dates before and after 2016-01-01.")

    # Select only feature columns (exclude date and target)
    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[target_col].copy()

    _assert_feature_matrix_schema(X_train, "X_train")
    _assert_feature_matrix_schema(X_test, "X_test")
    validate_feature_frame(X_train)
    validate_feature_frame(X_test)

    if progress_callback:
        progress_callback(35, "pipeline", "Building preprocessing and model pipeline...")

    pipeline = build_pipeline()

    if progress_callback:
        progress_callback(55, "fit", "Fitting model on training data...")
    pipeline.fit(X_train, y_train)

    if progress_callback:
        progress_callback(82, "evaluate", "Evaluating model on test split...")

    y_pred = pipeline.predict(X_test)
    metrics = {
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }

    if progress_callback:
        progress_callback(92, "save", "Saving champion model and metrics...")

    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    ref_stats = _build_reference_stats(frame[FEATURE_COLUMNS])
    with open(REFERENCE_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(ref_stats, f, indent=2)

    if progress_callback:
        progress_callback(98, "reference", "Saving reference statistics...")

    logger.info("Champion model saved to %s", MODEL_PATH)
    logger.info("Training complete | R2=%s | MAE=%s | RMSE=%s", metrics["r2"], metrics["mae"], metrics["rmse"])

    if progress_callback:
        progress_callback(100, "done", "Training completed successfully.")

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
