"""Drift detection module — compares new data statistics against the saved reference baseline."""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_STATS_PATH = PROJECT_ROOT / "models" / "reference_stats.json"

# A feature is considered "drifted" if its new mean deviates by more than this % of ref std
DRIFT_THRESHOLD_FACTOR = 0.5      # multiplier on reference std (more sensitive to catch practical 20-50% changes)
DRIFT_PERCENT_THRESHOLD = 0.20    # % of features that must drift to trigger global flag


def load_reference_stats() -> dict:
    """Load the reference baseline statistics saved at training time."""
    if not REFERENCE_STATS_PATH.exists():
        raise FileNotFoundError(
            "Reference stats not found. Train a model first to establish a baseline."
        )
    with open(REFERENCE_STATS_PATH) as f:
        return json.load(f)


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
    numeric_df = new_df.select_dtypes(include=[np.number])

    drift_report = {}
    drifted_features = []

    for col, stats in reference_stats.items():
        if col not in numeric_df.columns:
            drift_report[col] = {"status": "missing_in_new_data"}
            continue

        new_mean = float(numeric_df[col].mean())
        ref_mean = stats["mean"]
        ref_std = stats["std"] if stats["std"] > 0 else 1e-6   # avoid div-by-zero

        deviation = abs(new_mean - ref_mean)
        threshold = DRIFT_THRESHOLD_FACTOR * ref_std
        is_drifted = deviation > threshold

        drift_report[col] = {
            "reference_mean": round(ref_mean, 4),
            "new_mean": round(new_mean, 4),
            "deviation": round(deviation, 4),
            "threshold": round(threshold, 4),
            "drifted": is_drifted,
        }

        if is_drifted:
            drifted_features.append(col)

    total_tracked = len(reference_stats)
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
