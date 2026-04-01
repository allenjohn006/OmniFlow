"""Retraining module with champion-challenger promotion."""

from __future__ import annotations

import json
import logging
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing import FEATURE_COLUMNS, TARGET_DEFAULT, build_training_frame, validate_feature_frame
from src.training import MODEL_PATH, METRICS_PATH, MODELS_DIR, build_pipeline, get_champion_metrics

logger = logging.getLogger(__name__)

PROMOTION_THRESHOLD = 0.02

CHALLENGER_PATH = MODELS_DIR / "challenger.joblib"
CHALLENGER_METRICS_PATH = MODELS_DIR / "challenger_metrics.json"


def _assert_feature_matrix_schema(X: pd.DataFrame, frame_name: str) -> None:
    actual = list(X.columns)
    if actual != FEATURE_COLUMNS:
        raise ValueError(
            f"{frame_name} feature columns do not match expected schema. "
            f"Expected {FEATURE_COLUMNS}, got {actual}"
        )


def retrain_model(train_source, target_col: str = TARGET_DEFAULT) -> dict:
    """
    Train a challenger model on new data and compare it against the champion.

    If the challenger achieves a higher R² score, it becomes the new champion.

    Args:
        new_df (pd.DataFrame): New/incoming dataset.
        target_col (str): The target column name.

    Returns:
        dict with challenger metrics, champion metrics, and whether replacement occurred.
    """
    logger.info("Starting challenger model training...")

    frame = build_training_frame(train_source=train_source, target_col=target_col)
    # Use datetime comparison properly
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", target_col])
    train_mask = frame["date"] < pd.Timestamp("2016-01-01")
    train_df = frame[train_mask]
    test_df = frame[~train_mask]
    if train_df.empty or test_df.empty:
        raise ValueError("Temporal split failed for retraining dataset.")

    # Select only feature columns (exclude date and target)
    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[target_col].copy()

    _assert_feature_matrix_schema(X_train, "X_train")
    _assert_feature_matrix_schema(X_test, "X_test")
    validate_feature_frame(X_train)
    validate_feature_frame(X_test)

    challenger = build_pipeline()
    challenger.fit(X_train, y_train)
    y_pred = challenger.predict(X_test)

    challenger_metrics = {
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }

    joblib.dump(challenger, CHALLENGER_PATH)
    with open(CHALLENGER_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(challenger_metrics, f, indent=2)

    # Compare with champion
    champion_metrics = get_champion_metrics()
    champion_r2 = champion_metrics.get("r2", -9999)
    challenger_r2 = challenger_metrics.get("r2", -9999)

    replaced = False
    if challenger_r2 > champion_r2 + PROMOTION_THRESHOLD:
        logger.info(
            f"Challenger R²={challenger_r2} beats Champion R²={champion_r2} + {PROMOTION_THRESHOLD}. Replacing champion."
        )
        # Overwrite champion with challenger
        shutil.copy(CHALLENGER_PATH, MODEL_PATH)
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(challenger_metrics, f, indent=2)
        replaced = True
    else:
        logger.info(
            f"Champion R²={champion_r2} holds. Challenger R²={challenger_r2} not promoted."
        )

    return {
        "challenger_metrics": challenger_metrics,
        "champion_metrics": champion_metrics,
        "replaced": replaced,
        "message": (
            f"Challenger promoted! New champion R²={challenger_r2}"
            if replaced
            else f"Champion retained (R²={champion_r2}). Challenger R²={challenger_r2}"
        ),
    }
