"""Drift detection for Store Sales using statistical tests."""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Tuple

from scipy.stats import chisquare, ks_2samp

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_STATS_PATH = PROJECT_ROOT / "models" / "reference_stats.json"

NUMERICAL_PVALUE_THRESHOLD = 0.05
CATEGORICAL_PVALUE_THRESHOLD = 0.05
DRIFT_PERCENT_THRESHOLD = 0.20


def load_reference_stats() -> dict:
    """Load the reference baseline statistics saved at training time."""
    if not REFERENCE_STATS_PATH.exists():
        raise FileNotFoundError(
            "Reference stats not found. Train a model first to establish a baseline."
        )
    with open(REFERENCE_STATS_PATH) as f:
        return json.load(f)


def _ks_drift(new_series, ref_mean, ref_std):
    ref_std = max(ref_std, 1e-6)
    synthetic_ref = np.random.normal(loc=ref_mean, scale=ref_std, size=min(len(new_series), 5000))
    stat, pvalue = ks_2samp(new_series.values, synthetic_ref)
    return float(stat), float(pvalue)


def _chi_drift(new_series, ref_dist: dict):
    categories = sorted(set(new_series.astype(str).unique()) | set(ref_dist.keys()))
    observed = np.array([(new_series.astype(str) == c).sum() for c in categories], dtype=float)
    observed = observed / observed.sum() if observed.sum() > 0 else np.ones(len(categories)) / len(categories)
    expected = np.array([float(ref_dist.get(c, 0.0)) for c in categories], dtype=float)
    if expected.sum() == 0:
        expected = np.ones(len(categories)) / len(categories)
    else:
        expected = expected / expected.sum()

    # Scale to counts to apply chi-square goodness-of-fit.
    observed_counts = observed * 1000
    expected_counts = expected * 1000
    _, pvalue = chisquare(f_obs=observed_counts, f_exp=expected_counts)
    tvd = 0.5 * np.abs(observed - expected).sum()
    return float(pvalue), float(tvd)


def detect_drift(new_df, reference_stats: dict = None) -> Tuple[bool, dict]:
    """
    Detect data drift by comparing per-feature means against saved reference stats.

    Strategy: For each numeric column present in the reference, compute the
    absolute difference between the new mean and the reference mean. If that
    difference exceeds DRIFT_THRESHOLD_FACTOR × reference_std, the feature
    is flagged. Drift is declared globally when ≥ DRIFT_PERCENT_THRESHOLD of
    tracked features are flagged.

    Args:
        new_df (pd.DataFrame): Incoming production / new data.
        reference_stats (dict, optional): Pre-loaded stats. Loaded from disk if None.

    Returns:
        (drift_detected: bool, report: dict)
    """
    if reference_stats is None:
        reference_stats = load_reference_stats()

    import pandas as pd
    import pandas as pd

    numeric_df = new_df.select_dtypes(include=[np.number])
    categorical_df = new_df.select_dtypes(include=["object", "category"])

    drift_report = {}
    drifted_features = []

    numerical_ref = reference_stats.get("numerical", {})
    categorical_ref = reference_stats.get("categorical", {})

    for col, stats in numerical_ref.items():
        if col not in numeric_df.columns:
            drift_report[col] = {"status": "missing_in_new_data"}
            continue

        series = pd.to_numeric(numeric_df[col], errors="coerce").dropna()
        if series.empty:
            drift_report[col] = {"status": "empty_in_new_data"}
            continue

        ks_stat, pvalue = _ks_drift(series, stats["mean"], stats["std"])
        is_drifted = pvalue < NUMERICAL_PVALUE_THRESHOLD

        drift_report[col] = {
            "test": "ks_2samp",
            "reference_mean": round(float(stats["mean"]), 4),
            "new_mean": round(float(series.mean()), 4),
            "ks_stat": round(ks_stat, 4),
            "pvalue": round(pvalue, 6),
            "drifted": is_drifted,
        }

        if is_drifted:
            drifted_features.append(col)

    for col, dist in categorical_ref.items():
        if col not in categorical_df.columns:
            drift_report[col] = {"status": "missing_in_new_data"}
            continue

        pvalue, tvd = _chi_drift(categorical_df[col].fillna("Unknown"), dist)
        is_drifted = pvalue < CATEGORICAL_PVALUE_THRESHOLD
        drift_report[col] = {
            "test": "chi_square",
            "pvalue": round(pvalue, 6),
            "tvd": round(tvd, 6),
            "drifted": is_drifted,
        }
        if is_drifted:
            drifted_features.append(col)

    total_tracked = len(numerical_ref) + len(categorical_ref)
    drift_ratio = len(drifted_features) / total_tracked if total_tracked > 0 else 0
    drift_detected = drift_ratio >= DRIFT_PERCENT_THRESHOLD

    summary = {
        "drift_detected": drift_detected,
        "drifted_features": drifted_features,
        "drift_ratio": round(drift_ratio, 4),
        "drift_threshold_used": DRIFT_PERCENT_THRESHOLD,
        "feature_report": drift_report,
    }

    logger.info(
        f"Drift check: {len(drifted_features)}/{total_tracked} features drifted. "
        f"Global drift={'YES' if drift_detected else 'NO'}"
    )
    return drift_detected, summary
