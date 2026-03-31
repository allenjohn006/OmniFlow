"""Data preprocessing module — handles missing values, encoding, and train/test splitting."""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

PREPROCESSOR_PATH = MODELS_DIR / "preprocessors.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_columns.json"
REFERENCE_STATS_PATH = MODELS_DIR / "reference_stats.json"


def preprocess_data(df: pd.DataFrame, target_col: str, is_training: bool = True):
    """
    Preprocess the raw dataframe.

    - Drops rows where target is null.
    - Fills numeric NaNs with column mean.
    - Fills categorical NaNs with column mode.
    - Label-encodes all object/category columns.
    - During training: saves encoders + feature columns + reference stats.
    - During inference: loads saved encoders and reindexes columns.

    Args:
        df: Raw dataframe.
        target_col: Name of the target column.
        is_training: If True, fit+save encoders. If False, load saved encoders.

    Returns:
        X (pd.DataFrame), y (pd.Series or None if target absent)
    """
    df = df.copy()

    # ── Drop target NaNs ────────────────────────────────────────────────────
    if target_col in df.columns:
        logger.info(f"Original columns: {df.columns.tolist()}")
        df = df.dropna(subset=[target_col])
        y = df[target_col].reset_index(drop=True)
        df = df.drop(columns=[target_col])
        logger.info(f"After removing target '{target_col}': {df.columns.tolist()}")
    else:
        logger.info(f"Target col '{target_col}' not found. Available: {df.columns.tolist()}")
        y = None

    # ── Drop non-informative columns ─────────────────────────────────────────
    # Columns with >60% missing
    cols_to_drop = []
    for col in df.columns:
        if df[col].isnull().mean() > 0.6:
            cols_to_drop.append(col)
    
    # ── Handle Date column explicitly ────────────────────────────────────────
    # Convert datetime columns to ordinal BEFORE other processing
    date_cols = []
    for col in df.columns:
        if col not in cols_to_drop:
            # Check if column looks like a date
            if df[col].dtype == 'object' or 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors='coerce')
                    # If conversion worked and we have mostly non-null values, convert to ordinal
                    if df[col].notna().sum() / len(df) > 0.9:
                        logger.info(f"Converting {col} to ordinal")
                        df[col] = df[col].map(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
                        date_cols.append(col)
                except Exception as e:
                    logger.debug(f"Could not parse {col} as date: {e}")
    
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # ── Identify column types ────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ── Fill missing values ──────────────────────────────────────────────────
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    for col in cat_cols:
        mode_val = df[col].mode()
        df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")

    # ── Encode categorical columns ───────────────────────────────────────────
    encoders = {}
    if is_training:
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        # Save encoders
        joblib.dump(encoders, PREPROCESSOR_PATH)
        logger.info(f"Saved encoders for columns: {list(encoders.keys())}")

        # Save feature column order
        feature_cols = df.columns.tolist()
        logger.info(f"Saving feature columns (len={len(feature_cols)}): {feature_cols}")
        with open(FEATURE_COLS_PATH, "w") as f:
            json.dump(feature_cols, f)
        logger.info(f"✅ Saved feature columns to {FEATURE_COLS_PATH}")

        # ── Save reference stats for drift detection ─────────────────────────
        numeric_df = df[df.select_dtypes(include=[np.number]).columns]
        reference_stats = {}
        for col in numeric_df.columns:
            reference_stats[col] = {
                "mean": float(numeric_df[col].mean()),
                "std": float(numeric_df[col].std()),
                "min": float(numeric_df[col].min()),
                "max": float(numeric_df[col].max()),
            }
        with open(REFERENCE_STATS_PATH, "w") as f:
            json.dump(reference_stats, f, indent=2)
        logger.info("Saved reference statistics for drift detection.")

    else:
        # Load saved encoders
        if PREPROCESSOR_PATH.exists():
            encoders = joblib.load(PREPROCESSOR_PATH)
        for col in cat_cols:
            if col in encoders:
                le = encoders[col]
                # Handle unseen labels gracefully
                known_classes = set(le.classes_)
                df[col] = df[col].astype(str).map(
                    lambda x: x if x in known_classes else le.classes_[0]
                )
                df[col] = le.transform(df[col])
            else:
                df[col] = 0  # fallback for unknown new columns

        # Reorder / add missing columns to match training schema
        if FEATURE_COLS_PATH.exists():
            with open(FEATURE_COLS_PATH) as f:
                feature_cols = json.load(f)
            df = df.reindex(columns=feature_cols, fill_value=0)

    return df, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Split data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
