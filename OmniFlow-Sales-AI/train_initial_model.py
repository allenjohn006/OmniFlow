#!/usr/bin/env python
"""Script to train the initial champion model."""

import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import load_raw_data
from src.preprocessing import preprocess_data, split_data
from src.training import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    """Train initial model on old_data."""
    try:
        # Load data
        data_path = PROJECT_ROOT / "data" / "raw" / "old_data.csv"
        logger.info(f"Loading data from {data_path}...")
        df = load_raw_data(str(data_path))
        logger.info(f"📊 Loaded columns: {df.columns.tolist()}")
        
        target_col = "Units_Sold"
        logger.info(f"Using target column: {target_col}")
        
        # Preprocess
        logger.info("Preprocessing data...")
        X, y = preprocess_data(df, target_col=target_col, is_training=True)
        logger.info(f"✅ Preprocessed feature columns: {X.columns.tolist()}")
        logger.info(f"   Shape: {X.shape}")
        
        # Verify Marketing_Spend is present
        if "Marketing_Spend" not in X.columns:
            logger.warning("⚠️  WARNING: Marketing_Spend is MISSING from features!")
        if target_col in X.columns:
            logger.warning(f"⚠️  WARNING: {target_col} (TARGET) should NOT be in features!")
        
        # Split
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = split_data(X, y)
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logger.info(f"Train features: {X_train.columns.tolist()}")
        
        # Train
        logger.info("Training champion model...")
        metrics = train_model(X_train, X_test, y_train, y_test, model_name="champion")
        
        logger.info(f"✅ Model trained successfully!")
        logger.info(f"   R²: {metrics['r2']}")
        logger.info(f"   MAE: {metrics['mae']}")
        logger.info(f"   RMSE: {metrics['rmse']}")
        
    except Exception as e:
        logger.exception(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
