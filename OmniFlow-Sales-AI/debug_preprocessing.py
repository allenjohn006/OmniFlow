#!/usr/bin/env python
"""Debug script to trace preprocessing."""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import load_raw_data
from src.preprocessing import preprocess_data

# Load raw data
data_path = PROJECT_ROOT / "data" / "raw" / "old_data.csv"
print(f"\n📊 Original data from {data_path}:")
df = load_raw_data(str(data_path))
print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")
print(f"Data types:\n{df.dtypes}")

# Preprocess
target_col = "Units_Sold"
print(f"\n🔄 Preprocessing with target_col={target_col}...")
X, y = preprocess_data(df, target_col=target_col, is_training=True)

print(f"\n✅ Preprocessed features:")
print(f"X columns: {X.columns.tolist()}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Check saved files
import json
from pathlib import Path as P

saved_features_path = PROJECT_ROOT / "models" / "feature_columns.json"
saved_stats_path = PROJECT_ROOT / "models" / "reference_stats.json"

if saved_features_path.exists():
    with open(saved_features_path) as f:
        saved_features = json.load(f)
    print(f"\n💾 Saved feature_columns.json:")
    print(f"  {saved_features}")

if saved_stats_path.exists():
    with open(saved_stats_path) as f:
        saved_stats = json.load(f)
    print(f"\n💾 Saved reference_stats columns:")
    print(f"  {list(saved_stats.keys())}")
    
print("\n" + "="*60)
if "Marketing_Spend" not in saved_features:
    print("⚠️  WARNING: Marketing_Spend is MISSING from saved features!")
if "Units_Sold" in saved_features:
    print("⚠️  WARNING: Units_Sold (TARGET) should NOT be in features!")
