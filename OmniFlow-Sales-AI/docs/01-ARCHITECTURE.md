# 🏗️ System Architecture

## Overview

OmniFlow Sales AI is a production-grade MLOps platform built with a modern microservices-inspired architecture:
- **Backend**: FastAPI async REST API with background job processing
- **Frontend**: Django web framework with real-time AJAX progress tracking
- **ML Engine**: XGBoost regression model with scikit-learn preprocessing pipeline
- **Monitoring**: Statistical drift detection with automated retraining triggers

The system handles large-scale data (3M+ rows) efficiently through async processing, explicit dtype management for pandas 2.2+, and intelligent caching.

---

## 🎯 System Components

### 1. FastAPI Backend (`api/main.py`)

**Purpose**: Core ML operations and job orchestration  
**Port**: 8000

**Key Endpoints**:
```
POST  /train/start              Start async training job
GET   /train/status/{job_id}    Poll training progress
POST  /drift/start              Start drift detection job
GET   /drift/status/{job_id}    Poll drift progress
POST  /predict                  Real-time single prediction
GET   /health                   Health check
```

**Key Features**:
- **Async Job Architecture**: Uses threading with in-memory job stores (`TRAINING_JOBS`, `DRIFT_JOBS`)
- **Progress Callbacks**: 8-phase progress tracking during model training
- **Thread-Safe Operations**: Uses locks for concurrent access
- **JSON Serialization Safety**: `_json_safe()` converter handles numpy/pandas types

**Data Models**:
```python
Job State = {
    job_id: str,
    status: "queued" | "running" | "completed" | "failed",
    progress: int (0-100),
    stage: str (phase name),
    message: str (human-readable),
    started_at: ISO timestamp,
    updated_at: ISO timestamp,
    result: dict (metrics/recommendations),
    error: str (if failed)
}
```

---

### 2. Django Frontend (`django_app/omniapp/`)

**Purpose**: Web interface for training, drift analysis, and predictions  
**Port**: 8080

**Routes**:
```
GET  /                    Dashboard home
POST /upload/             Upload dataset and start training
GET  /upload_status/{job_id}/  Proxy to FastAPI status endpoint
GET  /train_result/       Display training results
POST /drift/              Upload data for drift detection
GET  /drift_status/{job_id}/   Proxy drift status
GET  /drift_result/       Display drift analysis report
GET  /predict/            Single prediction interface
```

**Key Features**:
- **AJAX Form Submission**: Non-blocking uploads with progress tracking
- **Live Progress Bar**: 2-second polling interval with animated UI
- **Auto-Redirect**: Redirects to results page on job completion
- **Error Handling**: Graceful error displays and resume capability

---

### 3. ML Pipeline (`src/`)

**Core Modules**:

#### `src/training.py`
- **Purpose**: Model training and metrics calculation
- **Key Functions**:
  - `build_pipeline()`: Creates scikit-learn Pipeline with preprocessing + XGBoost
  - `train_model()`: Trains model with progress callbacks
  - `get_champion_metrics()`: Returns baseline metrics (R², MAE, RMSE)
  - `load_champion_model()`: Loads persisted model from joblib
- **Outputs**: 
  - `models/champion.joblib` (trained model)
  - `models/champion_metrics.json` (baseline metrics)
  - `models/reference_stats.json` (feature statistics for drift detection)

#### `src/preprocessing.py`
- **Purpose**: Feature engineering and data validation
- **Key Functions**:
  - `build_training_frame()`: Merges multiple datasets, creates features, handles dtype casting
  - `validate_feature_frame()`: Ensures schema consistency
  - `extract_date_features()`: Year, month, day-of-week from dates
- **Features Engineered**:
  - Lag features (7-day historical)
  - Binary indicators (weekend, salary day)
  - Holiday type categorization
  - Cluster assignments from auxiliary data
- **Dtype Safety**: Explicit `float64` casting on all numerics (pandas 2.2+ requirement)

#### `src/drift.py`
- **Purpose**: Statistical drift detection using hypothesis testing
- **Strategy**: Compare new data distribution against reference training statistics
- **Tests**:
  - **Numerical Features**: Kolmogorov-Smirnov (KS) test
  - **Categorical Features**: Chi-square goodness-of-fit test
- **Outputs**:
  - Per-feature drift indicators (p-value, statistic)
  - Overall drift ratio (% drifted features)
  - Drift determination (boolean)

#### `src/retrain.py`
- **Purpose**: Automated retraining with challenger model evaluation
- **Flow**:
  1. Train challenger model on new data
  2. Evaluate challenger on new data and champion on new data
  3. Compare metrics (R², MAE, RMSE)
  4. Promote challenger if performance improves, else retain champion
- **Decision Logic**:
  - Retrain if: R² drops >5% OR MAE/RMSE increases >10% OR drift_ratio >75%
  - Otherwise: Retain champion

#### `src/ingestion.py`
- **Purpose**: Data loading with validation
- **Operations**:
  - Load CSV files from disk or bytes
  - Validate required columns
  - Handle missing values intelligently

#### `src/inference.py`
- **Purpose**: Single prediction endpoint
- **Operations**:
  - Load champion model
  - Validate input features
  - Generate prediction with confidence bounds

#### `src/utils.py`
- **Purpose**: Shared utility functions
- **Operations**:
  - Path resolution
  - File I/O helpers
  - Common data transformations

---

## 📊 Data Flow Diagrams

### Training Flow
```
User Uploads CSV
       ↓
Django receives file
       ↓
POST /train/start → FastAPI
       ↓
Background Thread Starts:
  1. Load & validate data
  2. Build training frame (feature engineering)
  3. Split: Train (pre-2016) | Test (post-2016)
  4. Create preprocessing pipeline
  5. Train XGBoost model with progress callbacks
  6. Calculate metrics (R², MAE, RMSE)
  7. Save model + reference stats + metrics
       ↓
Browser polls /train/status/{job_id} every 2s
       ↓
Job completes → Auto-redirect to results
```

### Drift Detection + Auto-Retrain Flow
```
User Uploads New Data
       ↓
Django receives file
       ↓
POST /drift/start → FastAPI
       ↓
Background Thread Starts:
  1. Load & validate new data
  2. Build new data frame (same features)
  3. Load reference stats from training
  4. Perform statistical drift tests (KS + Chi-square)
  5. Calculate % drifted features
  6. Evaluate champion on new data (get R², MAE, RMSE)
  7. Compare vs baseline metrics
       ↓
Decision Engine (Guardrails-based):
  IF (drift_detected AND performance_degraded) OR drift_ratio > 75%:
    → Start challenger training on new data
    → Compare challenger vs champion on new data
    → Promote if challenger better
  ELSE:
    → Retain champion (no retrain)
       ↓
Browser polls /drift/status/{job_id} every 2s
       ↓
Job completes with recommendation → Auto-redirect to results
```

### Prediction Flow
```
User enters features
       ↓
Django /predict/ form
       ↓
POST /predict → FastAPI
       ↓
FastAPI:
  1. Load champion model
  2. Validate feature schema
  3. Preprocess input (feature engineering, encoding)
  4. Generate prediction
  5. Return value + confidence bounds
       ↓
Django displays prediction
```

---

## 🔧 Key Design Patterns

### 1. **Async Job Architecture**
- **Problem**: Training on 3M+ rows blocks HTTP requests (timeout)
- **Solution**: Background thread with in-memory job state dictionary
- **Benefit**: Non-blocking UX with real-time progress updates

### 2. **Guardrail-Based Decision Logic**
- **Problem**: When should we retrain? Always? Never?
- **Solution**: Thresholds-based decision engine:
  - Performance thresholds (R² >5%, MAE/RMSE >10%)
  - Statistical thresholds (drift_ratio >75%)
  - Drift ≠ Performance (statistical drift doesn't always harm accuracy)
- **Benefit**: Prevents unnecessary retraining (model churn) while catching actual degradation

### 3. **Explicit Dtype Management**
- **Problem**: pandas 2.2+ promotes dtypes unexpectedly
- **Solution**: Explicit float64 casting at load time
- **Benefit**: Prevents sklearn pipeline crashes on unexpected types

### 4. **JSON Serialization Safety**
- **Problem**: numpy/pandas types not JSON-serializable
- **Solution**: Recursive `_json_safe()` converter before response()
- **Benefit**: No 500 errors on /status endpoints when returning complex objects

### 5. **Feature Schema Validation**
- **Problem**: Feature order mismatch between training and inference
- **Solution**: Explicit validation before model calls
- **Benefit**: Reproducible predictions across versions

---

## 📈 Scalability Considerations

**Current**: Handles 3M+ rows with 8 processes (n_jobs=-1 in XGBoost)

**Horizontal Scaling** (future):
- Move job stores to Redis (in-memory, distributed)
- Use Celery or similar for distributed job workers
- Containerize FastAPI for multi-instance Kubernetes deployment
- Use Postgres for persistent job history

**Vertical Scaling** (immediate):
- Increase XGBoost n_jobs for more CPU cores
- Use GPU offloading for XGBoost training
- Cache reference stats in memory (already done)

---

## 🔒 Security Considerations

**Current**:
- CORS middleware enabled (development mode)
- File uploads validated for CSV format
- Input schema validation
- No authentication (development environment)

**Production Hardening**:
- Add JWT authentication on FastAPI endpoints
- Restrict CORS to specific domains
- Implement file upload size limits
- Add rate limiting on /predict endpoint
- Use HTTPS/TLS for all endpoints
- Audit logging for model changes

