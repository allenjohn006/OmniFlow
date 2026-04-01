#!/usr/bin/env python3
"""Test script to verify dtype fix."""

from src.training import train_model

try:
    result = train_model()
    print(f"✓ Training successful!")
    print(f"  R² Score: {result['r2']}")
    print(f"  MAE: {result['mae']}")
    print(f"  RMSE: {result['rmse']}")
    print(f"  Train rows: {result['train_rows']}")
    print(f"  Test rows: {result['test_rows']}")
except Exception as e:
    print(f"✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
