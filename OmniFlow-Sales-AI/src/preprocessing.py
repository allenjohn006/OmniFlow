"""Shared preprocessing utilities for Store Sales training, inference, and retraining."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
# Primary data directory: the user's uploaded datasets live here
DATA_DIR = PROJECT_ROOT.parent / "data"
# Fallback to data/raw inside project if DATA_DIR doesn't exist
RAW_DIR = PROJECT_ROOT / "data" / "raw"

def _resolve_data_file(filename: str) -> Path:
    """Resolve a data file path, preferring the workspace-level data/ dir."""
    candidate = DATA_DIR / filename
    if candidate.exists():
        return candidate
    return RAW_DIR / filename

TRAIN_FILE = _resolve_data_file("train.csv")
STORES_FILE = _resolve_data_file("stores.csv")
OIL_FILE = _resolve_data_file("oil.csv")
HOLIDAYS_FILE = _resolve_data_file("holidays_events.csv")

TARGET_DEFAULT = "sales"

CATEGORICAL_FEATURES = ["family", "city", "state", "type", "holiday_type"]
NUMERICAL_FEATURES = [
    "store_nbr",
    "cluster",
    "onpromotion",
    "dcoilwtico",
    "lag_7",
    "year",
    "month",
    "dayofweek",
    "is_weekend",
    "is_salary_day",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def _read_csv_from_source(file_source) -> pd.DataFrame:
    if isinstance(file_source, bytes):
        return pd.read_csv(io.BytesIO(file_source))
    if isinstance(file_source, (str,)) or hasattr(file_source, "__fspath__"):
        return pd.read_csv(file_source)
    return pd.read_csv(file_source)


def _build_holiday_lookup(holidays_df: pd.DataFrame) -> pd.DataFrame:
    df = holidays_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "transferred" in df.columns:
        df = df[df["transferred"].astype(str).str.lower() != "true"]
    df["holiday_type"] = df.get("type", "None").fillna("None")
    lookup = (
        df.sort_values("date")
        .groupby("date", as_index=False)["holiday_type"]
        .first()
    )
    return lookup


def merge_store_sales_sources(train_df: pd.DataFrame) -> pd.DataFrame:
    """Merge train rows with store/oil/holiday side tables.

    Auxiliary files (stores, oil, holidays) are loaded from the resolved data
    directory. If any auxiliary file is missing, sensible defaults are filled in
    so the pipeline can still train without all side-tables.
    """
    df = train_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # --- stores.csv (city, state, type, cluster) ---
    stores_file = _resolve_data_file("stores.csv")
    if stores_file.exists():
        stores_df = pd.read_csv(stores_file)
        stores_df["store_nbr"] = pd.to_numeric(stores_df["store_nbr"], errors="coerce")
        df = df.merge(stores_df, on="store_nbr", how="left")
        logger.info("Merged stores.csv (%d rows)", len(stores_df))
    else:
        logger.warning("stores.csv not found at %s — using defaults", stores_file)
        for col in ["city", "state", "type"]:
            if col not in df.columns:
                df[col] = "Unknown"
        if "cluster" not in df.columns:
            df["cluster"] = 0

    # --- oil.csv (dcoilwtico) ---
    oil_file = _resolve_data_file("oil.csv")
    if oil_file.exists():
        oil_df = pd.read_csv(oil_file)
        oil_df["date"] = pd.to_datetime(oil_df["date"], errors="coerce")
        df = df.merge(oil_df[["date", "dcoilwtico"]], on="date", how="left")
        logger.info("Merged oil.csv (%d rows)", len(oil_df))
    else:
        logger.warning("oil.csv not found at %s — setting dcoilwtico=0.0", oil_file)
        if "dcoilwtico" not in df.columns:
            df["dcoilwtico"] = 0.0

    # --- holidays_events.csv (holiday_type) ---
    holidays_file = _resolve_data_file("holidays_events.csv")
    if holidays_file.exists():
        holidays_df = pd.read_csv(holidays_file)
        holiday_lookup = _build_holiday_lookup(holidays_df)
        df = df.merge(holiday_lookup, on="date", how="left")
        logger.info("Merged holidays_events.csv (%d rows)", len(holidays_df))
    else:
        logger.warning("holidays_events.csv not found at %s — setting holiday_type=None", holidays_file)
        if "holiday_type" not in df.columns:
            df["holiday_type"] = "None"

    df["holiday_type"] = df["holiday_type"].fillna("None")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["dayofweek"] = out["date"].dt.dayofweek
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    out["is_salary_day"] = out["date"].dt.day.isin([15, 30]).astype(int)
    return out


def add_lag_feature(
    df: pd.DataFrame,
    fallback_lag_7: Optional[float] = None,
    target_col: str = "sales",
) -> pd.DataFrame:
    out = df.copy()
    if "lag_7" in out.columns:
        out["lag_7"] = pd.to_numeric(out["lag_7"], errors="coerce")
        return out

    # Use the actual target column to build lag features
    lag_source = target_col if target_col in out.columns else "sales"
    if lag_source in out.columns:
        out = out.sort_values(["store_nbr", "family", "date"])
        out["lag_7"] = out.groupby(["store_nbr", "family"])[lag_source].shift(7)
    else:
        out["lag_7"] = fallback_lag_7 if fallback_lag_7 is not None else 0.0

    if fallback_lag_7 is None:
        fallback_lag_7 = 0.0
    out["lag_7"] = out["lag_7"].fillna(fallback_lag_7)
    return out


def normalize_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = "Unknown"
        out[col] = out[col].astype(str).fillna("Unknown")

    for col in NUMERICAL_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    out["dcoilwtico"] = out["dcoilwtico"].ffill().bfill().fillna(0.0)
    out[NUMERICAL_FEATURES] = out[NUMERICAL_FEATURES].fillna(0.0).astype("float64")
    return out


def validate_feature_frame(df: pd.DataFrame) -> None:
    """Validate feature schema before model fit/predict to fail early with clear errors."""
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Feature frame is missing required columns: {missing}")

    datetime_features = [col for col in FEATURE_COLUMNS if is_datetime64_any_dtype(df[col])]
    if datetime_features:
        raise ValueError(
            f"Datetime columns found in model features: {datetime_features}. "
            "Ensure `date` is excluded and only engineered numeric/categorical features are used."
        )

    non_numeric = [col for col in NUMERICAL_FEATURES if not is_numeric_dtype(df[col])]
    if non_numeric:
        raise ValueError(f"Non-numeric dtypes found in numerical features: {non_numeric}")


def build_training_frame(train_source=None, target_col: str = TARGET_DEFAULT) -> pd.DataFrame:
    """
    Build merged and feature-engineered training frame.

    `train_source` can be None (loads from workspace data/ directory),
    a file path, or uploaded bytes from the web UI.
    Auxiliary files (stores.csv, oil.csv, holidays_events.csv) are
    auto-resolved; missing files are handled with sensible defaults.
    """
    if train_source is None:
        train_file = _resolve_data_file("train.csv")
        train_df = pd.read_csv(train_file)
        logger.info("Loaded training data from %s (%d rows)", train_file, len(train_df))
    else:
        train_df = _read_csv_from_source(train_source)
        logger.info("Loaded training data from upload (%d rows)", len(train_df))

    required = {"date", "store_nbr", "family", "onpromotion"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"Training data missing required columns: {sorted(missing)}")
    if target_col not in train_df.columns:
        available = sorted(train_df.columns.tolist())
        raise ValueError(
            f"Target column '{target_col}' not found in training data. "
            f"Available columns: {available}"
        )

    df = merge_store_sales_sources(train_df)
    df = add_time_features(df)
    df = add_lag_feature(df, target_col=target_col)
    df = normalize_feature_types(df)
    df = df.dropna(subset=[target_col, "date"])
    validate_feature_frame(df)
    return df


def build_inference_frame(payload: Dict) -> pd.DataFrame:
    """Build a single-row inference frame without joining raw CSV files."""
    df = pd.DataFrame([payload])
    required = {"date", "store_nbr", "family", "onpromotion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction payload missing required fields: {sorted(missing)}")

    df = add_time_features(df)
    df = add_lag_feature(df, fallback_lag_7=float(payload.get("lag_7", 0.0)))
    df = normalize_feature_types(df)
    validate_feature_frame(df)
    return df
