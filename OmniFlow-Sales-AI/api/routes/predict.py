"""Prediction API routes."""

from fastapi import APIRouter

router = APIRouter()

@router.post("/predict")
def predict(data: dict):
    """Make predictions."""
    pass
