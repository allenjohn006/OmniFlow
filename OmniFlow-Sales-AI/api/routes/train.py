"""Training API routes."""

from fastapi import APIRouter

router = APIRouter()

@router.post("/train")
def train():
    """Train the model."""
    pass
