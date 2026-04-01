# OmniFlow Sales AI

Store Sales MLOps backend built with FastAPI, Django, scikit-learn Pipeline, and XGBoost.

## Refactor Update (April 2026)

This project was refactored to fully support the Kaggle Favorita Store Sales dataset and remove old Ecommerce-specific logic.

### What Was Implemented

1. Shared preprocessing for train, predict, and retrain
- Single feature engineering flow in src/preprocessing.py
- Date features: year, month, dayofweek, is_weekend, is_salary_day
- Lag support: lag_7 is computed during training/retraining and accepted from payload for prediction

2. Unified sklearn Pipeline (no manual encoders)
- ColumnTransformer + OneHotEncoder + XGBoost in one Pipeline
- No separate LabelEncoder persistence
- One model artifact used for inference

3. Dataset-native training flow
- Training merges train.csv + stores.csv + oil.csv + holidays_events.csv
- Temporal split used:
  - Train: date < 2016-01-01
  - Test: date >= 2016-01-01

4. Lightweight prediction API
- Prediction does not merge raw CSVs
- API consumes prepared request fields directly, including lag_7

5. Retraining consistency
- Challenger uses the exact same preprocessing + pipeline structure as champion
- Champion promotion only when challenger R2 > champion R2 + 0.02

6. Drift detection modernization
- Numerical drift: KS test
- Categorical drift: chi-square with TVD reporting

### Latest Verified Training Metrics

- R2: 0.882
- MAE: 102.8503
- RMSE: 452.5207
- Train rows: 1,945,944
- Test rows: 1,054,944

## Current Folder Expectations

### Raw Data
Place these files in data/raw:
- train.csv
- stores.csv
- oil.csv
- holidays_events.csv

### Active Model Artifacts
models now keeps:
- champion.joblib
- champion_metrics.json
- reference_stats.json

## API Endpoints (Current)

- GET /health
- POST /train
- POST /predict
- POST /predict-legacy
- POST /drift-retrain

### /train Behavior
- If file is uploaded: trains from uploaded train.csv content
- If no file is uploaded: trains from data/raw/train.csv
- Always uses local side tables from data/raw for merges

### /predict Payload (Recommended)

{
  "date": "2017-08-10",
  "store_nbr": 1,
  "family": "GROCERY I",
  "onpromotion": 5,
  "city": "Quito",
  "state": "Pichincha",
  "type": "D",
  "cluster": 13,
  "dcoilwtico": 48.5,
  "holiday_type": "None",
  "lag_7": 1200
}

Notes:
- lag_7 should be provided by caller if possible.
- If lag_7 is omitted, backend falls back to 0.0.

## Files Added / Refactored

### Added
- src/inference.py

### Refactored
- src/preprocessing.py
- src/training.py
- src/retrain.py
- src/drift.py
- api/main.py
- dev.py
- django_app/omniapp/views.py
- requirements.txt

## Legacy Items Removed

Removed old and unused components:
- train_initial_model.py
- debug_preprocessing.py
- install_deps.ps1
- start.bat
- api/routes/* placeholder files
- models/champion_model.pkl
- models/preprocessors.pkl
- models/feature_columns.json
- models/sales_model.joblib

## Run Instructions

From OmniFlow-Sales-AI folder:

1. Install dependencies
- python -m pip install -r requirements.txt

2. Train model
- python dev.py run-train

3. Run API only
- python dev.py run-api

4. Run API + Django UI
- python dev.py run-all

## Notes

- This codebase now assumes Store Sales schema and artifacts.
- The old Ecommerce pipeline and manual encoding flow are no longer part of runtime logic.
- If mlflow.db cannot be deleted on Windows, close the process locking it and retry deletion.
