"""Retraining module — champion vs challenger model comparison and safe replacement."""

import joblib
import json
import logging
import shutil
from pathlib import Path

from src.preprocessing import preprocess_data, split_data
from src.training import train_model, get_champion_metrics, MODEL_PATH, METRICS_PATH, MODELS_DIR

logger = logging.getLogger(__name__)

CHALLENGER_PATH = MODELS_DIR / "challenger_model.pkl"
CHALLENGER_METRICS_PATH = MODELS_DIR / "challenger_metrics.json"


def retrain_model(new_df, target_col: str) -> dict:
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

    # Preprocess new data (fit=False so we don't overwrite reference stats)
    X, y = preprocess_data(new_df, target_col=target_col, is_training=False)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train challenger
    challenger_metrics = train_model(X_train, X_test, y_train, y_test, model_name="challenger")

    # Save challenger separately
    challenger_src = MODELS_DIR / "challenger_model.pkl"
    with open(CHALLENGER_METRICS_PATH, "w") as f:
        json.dump(challenger_metrics, f, indent=2)

    # Compare with champion
    champion_metrics = get_champion_metrics()
    champion_r2 = champion_metrics.get("r2", -9999)
    challenger_r2 = challenger_metrics.get("r2", -9999)

    replaced = False
    if challenger_r2 > champion_r2:
        logger.info(
            f"Challenger R²={challenger_r2} beats Champion R²={champion_r2}. Replacing champion."
        )
        # Overwrite champion with challenger
        shutil.copy(challenger_src, MODEL_PATH)
        with open(METRICS_PATH, "w") as f:
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
