"""Data drift API routes."""

from fastapi import APIRouter

router = APIRouter()

@router.get("/drift")
def check_drift():
    """Check for data drift."""
    pass
