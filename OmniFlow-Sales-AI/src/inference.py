"""Inference helpers for lightweight prediction requests."""

from __future__ import annotations

from typing import Any, Dict

from src.preprocessing import FEATURE_COLUMNS, build_inference_frame
from src.training import load_champion_model


def predict_from_payload(payload: Dict[str, Any]) -> float:
    """Generate prediction from a lightweight JSON payload."""
    model = load_champion_model()
    frame = build_inference_frame(payload)
    X = frame[FEATURE_COLUMNS]
    pred = model.predict(X)[0]
    return float(pred)
