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
RAW_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_FILE = RAW_DIR / "train.csv"
STORES_FILE = RAW_DIR / "stores.csv"
OIL_FILE = RAW_DIR / "oil.csv"
HOLIDAYS_FILE = RAW_DIR / "holidays_events.csv"

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
    """Merge train rows with store/oil/holiday side tables using local raw data files."""
    stores_df = pd.read_csv(STORES_FILE)
    oil_df = pd.read_csv(OIL_FILE)
    holidays_df = pd.read_csv(HOLIDAYS_FILE)

    df = train_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    stores_df["store_nbr"] = pd.to_numeric(stores_df["store_nbr"], errors="coerce")
    oil_df["date"] = pd.to_datetime(oil_df["date"], errors="coerce")

    holiday_lookup = _build_holiday_lookup(holidays_df)

    df = df.merge(stores_df, on="store_nbr", how="left")
    df = df.merge(oil_df[["date", "dcoilwtico"]], on="date", how="left")
    df = df.merge(holiday_lookup, on="date", how="left")
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


def add_lag_feature(df: pd.DataFrame, fallback_lag_7: Optional[float] = None) -> pd.DataFrame:
    out = df.copy()
    if "lag_7" in out.columns:
        out["lag_7"] = pd.to_numeric(out["lag_7"], errors="coerce")
        return out

    if "sales" in out.columns:
        out = out.sort_values(["store_nbr", "family", "date"])
        out["lag_7"] = out.groupby(["store_nbr", "family"])["sales"].shift(7)
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

    `train_source` can be None (use data/raw/train.csv), file path, or uploaded bytes.
    """
    if train_source is None:
        train_df = pd.read_csv(TRAIN_FILE)
    else:
        train_df = _read_csv_from_source(train_source)

    required = {"date", "store_nbr", "family", "onpromotion"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"Training data missing required columns: {sorted(missing)}")
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data")

    df = merge_store_sales_sources(train_df)
    df = add_time_features(df)
    df = add_lag_feature(df)
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
