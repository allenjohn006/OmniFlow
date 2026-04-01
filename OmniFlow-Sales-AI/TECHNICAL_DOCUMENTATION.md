# 🔧 Technical Documentation: Async Training & Drift Detection

This document provides detailed technical information about the async training architecture, drift detection system, and dtype handling implemented in the TimeSeries branch.

---

## Table of Contents

1. [Async Training Architecture](#async-training-architecture)
2. [Drift Detection System](#drift-detection-system)
3. [Dtype Safety (Pandas 2.2+ Compatibility)](#dtype-safety)
4. [Progress Tracking & UI](#progress-tracking--ui)
5. [API Reference](#api-reference)
6. [Database Schema](#database-schema)
7. [Performance Considerations](#performance-considerations)

---

## Async Training Architecture

### Overview

The async training system eliminates the 120+ second request timeout by moving long-running model training to background threads. The user receives a job ID immediately and polls for progress.

### Flow Diagram

```
┌─────────────────┐
│  Django Form    │
└────────┬────────┘
         │ AJAX POST /upload
         │ (XMLHttpRequest header)
         ▼
┌──────────────────────────┐
│ Django views.upload()    │ (detects AJAX)
└────────┬─────────────────┘
         │ requests.post() /train/start
         │ (FastAPI backend)
         ▼
┌──────────────────────────┐      ┌──────────────────────┐
│ FastAPI POST /train/start│      │ Background Thread    │
│  - Generate job_id       │      │ _run_training_job()  │
│  - Spawn background ───┐ │      │  - Load data         │
│  - Return job_id       │ └─────►│  - Preprocess        │
└──────────────────────────┘      │  - Train model       │
         │                        │  - Save checkpoint   │
         │ HTTP 200 OK            │  - Update job_state  │
         │ {job_id, poll_url}     └──────────────────────┘
         │                               ▲
         ▼                               │ (progress_callback)
┌──────────────────────────┐            │
│ Browser polls            │────────────┘ Every 2s:
│ /upload/status/{job_id}/ │ GET /train/status/{job_id}
│  - Updates progress %    │
│  - Updates stage name    │
│  - Updates message       │
│  - Updates progress bar  │
└──────────────────────────┘
         │
         │ On completion (status=="completed")
         ▼
┌──────────────────────────┐
│ Auto-redirect to         │
│ /train/?job_id={job_id}  │
└──────────────────────────┘
         ▼
┌──────────────────────────┐
│ Results page with        │
│ metrics (R², MAE, RMSE)  │
└──────────────────────────┘
```

### Implementation Details

#### 1. Job State Management (`api/main.py`)

```python
from threading import Lock
from typing import Dict
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class TrainingJob:
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: int  # 0-100
    stage: str  # descriptive stage name
    message: str  # human-readable message
    result: dict = None  # metrics on completion
    error: str = None  # error message on failure
    created_at: str = None
    updated_at: str = None

# Global in-memory job storage
TRAINING_JOBS: Dict[str, dict] = {}
TRAINING_JOBS_LOCK = Lock()

def _update_job(job_id: str, **updates):
    """Thread-safe job state update"""
    with TRAINING_JOBS_LOCK:
        if job_id in TRAINING_JOBS:
            TRAINING_JOBS[job_id].update({
                **updates,
                'updated_at': datetime.utcnow().isoformat() + 'Z'
            })

def _now_iso() -> str:
    """ISO 8601 timestamp"""
    return datetime.utcnow().isoformat() + 'Z'
```

#### 2. Background Training Job (`api/main.py`)

```python
def _run_training_job(job_id: str, train_file: str, target_col: str):
    """Runs in background thread, updates job state via callbacks"""
    
    def progress_callback(percent: int, stage: str, message: str):
        """Called by train_model() at key phases"""
        _update_job(job_id, 
                    progress=percent, 
                    stage=stage, 
                    message=message)
    
    try:
        _update_job(job_id, status='processing')
        
        # Call training with progress callback
        metrics = train_model(
            train_file=train_file,
            target_col=target_col,
            progress_callback=progress_callback
        )
        
        _update_job(job_id, 
                    status='completed', 
                    progress=100, 
                    stage='done',
                    message='Training complete!',
                    result=metrics)
    
    except Exception as e:
        _update_job(job_id, 
                    status='failed', 
                    error=str(e))

@app.post("/train/start")
async def start_training(file: UploadFile, target_col: str):
    """Non-blocking training endpoint"""
    
    # Save temp file
    temp_file = f"/tmp/{file.filename}"
    with open(temp_file, 'wb') as f:
        f.write(await file.read())
    
    # Create job entry
    job_id = str(uuid.uuid4())
    with TRAINING_JOBS_LOCK:
        TRAINING_JOBS[job_id] = {
            'status': 'accepted',
            'progress': 0,
            'stage': 'queued',
            'message': 'Waiting to start training...',
            'created_at': _now_iso(),
            'updated_at': _now_iso()
        }
    
    # Spawn background thread (daemon=True = exits when main process exits)
    threading.Thread(
        target=_run_training_job,
        args=(job_id, temp_file, target_col),
        daemon=True
    ).start()
    
    return {
        'status': 'accepted',
        'job_id': job_id,
        'poll_url': f'/train/status/{job_id}'
    }

@app.get("/train/status/{job_id}")
async def get_training_status(job_id: str):
    """Status polling endpoint"""
    
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get(job_id)
    
    if not job:
        return {'status': 'not_found'}, 404
    
    return job
```

#### 3. Django Proxy Endpoint (`django_app/omniapp/views.py`)

```python
def upload_status(request, job_id):
    """Proxy FastAPI status to browser (avoids CORS issues)"""
    
    try:
        response = requests.get(
            f"{FASTAPI_URL}/train/status/{job_id}",
            timeout=20
        )
        return JsonResponse(response.json())
    
    except requests.ConnectionError:
        return JsonResponse(
            {'error': 'Backend connection failed'},
            status=503
        )
    except Exception as e:
        return JsonResponse(
            {'error': str(e)},
            status=500
        )

def upload(request):
    """AJAX-aware training trigger"""
    
    if request.method == 'POST':
        # Detect AJAX request
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        
        file = request.FILES.get('file')
        target_col = request.POST.get('target_col')
        
        if is_ajax:
            # Non-blocking: call FastAPI /train/start
            response = requests.post(
                f"{FASTAPI_URL}/train/start",
                files={'file': file},
                data={'target_col': target_col},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return JsonResponse({
                    'status': 'accepted',
                    'job_id': data['job_id'],
                    'poll_url': f'/upload/status/{data["job_id"]}/'
                })
        
        else:
            # Non-AJAX fallback: legacy synchronous flow
            # (not used in new flow, kept for compatibility)
            pass
```

---

## Drift Detection System

### Overview

Drift detection compares feature distributions between:
- **Reference**: Training data statistics (stored in `models/reference_stats.json`)
- **Current**: New data being checked

### Statistical Method

#### Z-Score Drift Detection

```python
def detect_drift(df_current, reference_stats, threshold=2.0):
    """
    Z-score method: measures how many standard deviations 
    the new mean is from the reference mean
    
    z = (μ_new - μ_ref) / σ_ref
    
    If |z| > 2: Feature has drifted (95% confidence)
    """
    
    drift_report = {
        'features': {},
        'total_features': 0,
        'drifted_features': 0,
        'drift_ratio': 0.0
    }
    
    for feature, ref_stats in reference_stats.items():
        ref_mean = ref_stats['mean']
        ref_std = ref_stats['std']
        
        current_mean = df_current[feature].mean()
        
        # Calculate z-score
        if ref_std > 0:
            z_score = abs((current_mean - ref_mean) / ref_std)
        else:
            z_score = 0
        
        is_drifted = z_score > threshold
        
        drift_report['features'][feature] = {
            'reference_mean': ref_mean,
            'current_mean': current_mean,
            'z_score': z_score,
            'deviation_percent': ((current_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else 0,
            'is_drifted': is_drifted
        }
        
        drift_report['total_features'] += 1
        if is_drifted:
            drift_report['drifted_features'] += 1
    
    drift_report['drift_ratio'] = (
        drift_report['drifted_features'] / 
        drift_report['total_features']
    ) if drift_report['total_features'] > 0 else 0
    
    return drift_report
```

### Implementation (`src/drift.py`)

```python
import json
from pathlib import Path
import pandas as pd

REFERENCE_STATS_PATH = 'models/reference_stats.json'

def load_reference_stats():
    """Load baseline statistics from training phase"""
    with open(REFERENCE_STATS_PATH, 'r') as f:
        return json.load(f)

def save_reference_stats(df):
    """Save reference statistics after training"""
    stats = {}
    
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'count': int(df[col].count())
        }
    
    with open(REFERENCE_STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

def check_drift(file_path: str, target_col: str, threshold: float = 2.0) -> dict:
    """
    Main drift detection function
    
    Args:
        file_path: CSV file to check
        target_col: Target variable column name
        threshold: Z-score threshold (default 2.0 = 95% confidence)
    
    Returns:
        {
            'drift_ratio': 0.73,
            'drifted_count': 11,
            'total_count': 15,
            'retrain_trigger': True,
            'features': {
                'feature_name': {
                    'reference_mean': 100.5,
                    'current_mean': 120.3,
                    'z_score': 2.15,
                    'is_drifted': True
                }
            }
        }
    """
    
    # Load reference baseline
    reference_stats = load_reference_stats()
    
    # Load new data
    df_new = pd.read_csv(file_path)
    
    # Remove target column for drift analysis
    if target_col in df_new.columns:
        df_new = df_new.drop(columns=[target_col])
    
    # Detect drift
    report = detect_drift(df_new, reference_stats, threshold)
    
    # Auto-trigger retrain if drift > 75%
    report['retrain_trigger'] = report['drift_ratio'] > 0.75
    
    return report
```

### FastAPI Endpoint (`api/main.py`)

```python
@app.post("/drift")
async def check_for_drift(
    file: UploadFile,
    target_col: str = "sales",
    threshold: float = 2.0
):
    """Check for data drift in uploaded CSV"""
    
    # Save temp file
    temp_file = f"/tmp/drift_{file.filename}"
    with open(temp_file, 'wb') as f:
        f.write(await file.read())
    
    try:
        # Run drift detection
        report = check_drift(temp_file, target_col, threshold)
        
        return {
            'status': 'completed',
            'drift_ratio': report['drift_ratio'],
            'drifted_count': report['drifted_features'],
            'total_count': report['total_features'],
            'retrain_trigger': report['retrain_trigger'],
            'features': report['features']
        }
    
    except Exception as e:
        return {'error': str(e)}, 400
    
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
```

---

## Dtype Safety

### Problem: Pandas 2.2+ Dtype Promotion

**Symptom**: `TypeError: Cannot cast from Int64DType to datetime64`

**Root Cause**:
- Pandas 2.2 introduced nullable `Int64DType` for integer columns
- When nullable columns mixed with temporal types, scikit-learn's ColumnTransformer crashes
- Example:
  ```python
  # This causes the crash:
  df = pd.DataFrame({
      'year': pd.array([2013, 2014], dtype='Int64'),  # Nullable!
      'date': pd.to_datetime([...])  # Datetime!
  })
  ```

### Solution: Explicit Float64 Casting

In `src/preprocessing.py`:

```python
def normalize_feature_types(df):
    """
    Convert all numerical features to explicit float64
    
    Purpose: Avoid pandas 2.2+ nullable Int64 promotion issues
    """
    
    out = df.copy()
    
    # Ensure all numerical columns are float64
    NUMERICAL_FEATURES = [
        'store_nbr', 'cluster', 'onpromotion', 'dcoilwrico', 
        'lag_7', 'year', 'month', 'dayofweek'
    ]
    
    for col in NUMERICAL_FEATURES:
        if col in out.columns:
            # Step 1: Coerce to numeric (handles strings, NaN, etc.)
            out[col] = pd.to_numeric(out[col], errors='coerce')
            
            # Step 2: Fill missing values
            out[col].fillna(0.0, inplace=True)
            
            # Step 3: EXPLICIT dtype conversion to float64
            out[col] = out[col].astype('float64')
    
    # Ensure batch fill for any remaining NaN
    out[NUMERICAL_FEATURES] = out[NUMERICAL_FEATURES].fillna(0.0).astype('float64')
    
    return out
```

### Validation Helper

```python
def validate_feature_frame(df, frame_name="DataFrame"):
    """
    Assert that feature matrix has correct dtypes
    
    Checks:
    - No datetime64 columns (should be dropped before model input)
    - All numeric columns are float64
    - No NaN values
    """
    
    # Check for datetime columns (should have been dropped)
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols:
        raise ValueError(
            f"{frame_name} contains datetime columns: {datetime_cols}. "
            f"Drop date columns before model input."
        )
    
    # Check for nullable Int64
    int64_cols = [col for col in df.columns 
                  if hasattr(df[col].dtype, 'name') 
                  and df[col].dtype.name == 'Int64']
    if int64_cols:
        raise ValueError(
            f"{frame_name} contains nullable Int64 columns: {int64_cols}. "
            f"Convert to float64 explicitly."
        )
    
    # Check for NaN
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        raise ValueError(
            f"{frame_name} contains {nan_count} NaN values. "
            f"All values must be filled."
        )
```

### Integration in Training Pipeline

```python
def train_model(train_file, test_file, target_col, progress_callback=None):
    """Full training with dtype safety checks"""
    
    if progress_callback:
        progress_callback(5, "Loading", "Loading data...")
    
    # Load raw data
    X_train = pd.read_csv(train_file)
    
    # Preprocess with dtype normalization
    X_train = normalize_feature_types(X_train)
    
    # Validate before model input
    validate_feature_frame(X_train, "X_train")
    
    if progress_callback:
        progress_callback(35, "Pipeline", "Building feature transformer...")
    
    # Now safe to build pipeline (no dtype issues)
    pipeline = Pipeline([...])
    pipeline.fit(X_train, y_train)
    
    return metrics
```

---

## Progress Tracking & UI

### Backend Progress Callback System

**Design Pattern**: Functional callback passed to train_model()

```python
# Instead of:
def train_model():
    print("Loading data...")  # Hard to integrate UI
    # ...

# We use:
def train_model(progress_callback=None):
    if progress_callback:
        progress_callback(5, "Loading", "Loading data...")
    # ...
```

### Progress Phases

```
Phase 1 (5%):   "Loading"    → "Loading data..."
Phase 2 (20%):  "Split"      → "Splitting train/test..."
Phase 3 (35%):  "Pipeline"   → "Building feature transformer..."
Phase 4 (55%):  "Training"   → "Fitting XGBoost (300 trees)..."
Phase 5 (82%):  "Evaluate"   → "Calculating metrics..."
Phase 6 (92%):  "Saving"     → "Saving model to disk..."
Phase 7 (98%):  "Reference"  → "Computing reference statistics..."
Phase 8 (100%): "done"       → "Training complete!"
```

### Frontend Progress Polling (`upload.html`)

```javascript
function pollStatus(jobId) {
    fetch(`/upload/status/${jobId}/`)
        .then(resp => resp.json())
        .then(payload => {
            // Update progress bar
            updateProgress(payload);
            
            // Check if complete
            if (payload.status === 'completed') {
                // Redirect to results page
                window.location.href = `/train/?job_id=${jobId}`;
                return;
            }
            
            if (payload.status === 'failed') {
                // Show error
                showError(payload.error);
                return;
            }
            
            // Poll again in 2 seconds
            setTimeout(() => pollStatus(jobId), 2000);
        });
}

function updateProgress(payload) {
    const percent = payload.progress || 0;
    const stage = payload.stage || 'Unknown';
    const message = payload.message || '';
    
    // Update UI elements
    document.getElementById('training-percent').textContent = `${percent}%`;
    document.getElementById('training-stage').textContent = `Stage: ${stage}`;
    document.getElementById('training-message').textContent = message;
    
    // Update progress bar width
    document.getElementById('progress-fill').style.width = `${percent}%`;
    
    // Show training panel
    document.querySelector('.training-panel').classList.add('show');
}
```

### CSS Styling

```css
.training-panel {
    background: linear-gradient(135deg, #2c3e50 0%, #1a3a40 100%);
    border-radius: 12px;
    padding: 1rem;
    display: none;
    margin: 1rem 0;
}

.training-panel.show {
    display: block;
    animation: slideIn 0.3s ease;
}

.progress-track {
    height: 10px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.1);
    overflow: hidden;
    margin: 1rem 0;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366f1 0%, #10b981 100%);
    width: 0%;
    transition: width 0.35s ease;
    border-radius: 999px;
}
```

---

## API Reference

### Authentication
All endpoints are currently public (no authentication required).

### Endpoints

#### `/train/start` (POST)
Start async model training.

**Request**:
```json
{
  "file": "<binary CSV>",
  "target_col": "sales"
}
```

**Response** (200):
```json
{
  "status": "accepted",
  "job_id": "89841818-cfb9-4b64-bccf-5ab37d9820dd",
  "poll_url": "/train/status/89841818-cfb9-4b64-bccf-5ab37d9820dd"
}
```

---

#### `/train/status/{job_id}` (GET)
Poll training job progress.

**Response** (200) - In Progress:
```json
{
  "status": "processing",
  "progress": 55,
  "stage": "Training",
  "message": "Fitting XGBoost (300 trees)...",
  "created_at": "2026-04-01T10:30:00Z",
  "updated_at": "2026-04-01T10:31:15Z"
}
```

**Response** (200) - Completed:
```json
{
  "status": "completed",
  "progress": 100,
  "stage": "done",
  "message": "Training complete!",
  "result": {
    "r2_score": 0.882,
    "mae": 102.85,
    "rmse": 452.52,
    "train_samples": 1000000,
    "test_samples": 500000
  },
  "created_at": "2026-04-01T10:30:00Z",
  "updated_at": "2026-04-01T10:32:35Z"
}
```

**Response** (404) - Job Not Found:
```json
{
  "status": "not_found"
}
```

---

#### `/drift` (POST)
Check for data drift in uploaded CSV.

**Request**:
```json
{
  "file": "<binary CSV>",
  "target_col": "sales",
  "threshold": 2.0
}
```

**Response** (200):
```json
{
  "status": "completed",
  "drift_ratio": 0.7333,
  "drifted_count": 11,
  "total_count": 15,
  "retrain_trigger": true,
  "features": {
    "year": {
      "reference_mean": 2014.8379,
      "current_mean": 2016.3834,
      "z_score": 12.45,
      "deviation_percent": 0.11,
      "is_drifted": true
    },
    "family": {
      "reference_mean": 5.12,
      "current_mean": 5.10,
      "z_score": 0.15,
      "deviation_percent": -0.39,
      "is_drifted": false
    }
  }
}
```

---

## Database Schema

### Currently In-Memory (for persistence, add Redis/PostgreSQL)

```python
# Python dict structure
{
    "job_id": {
        "job_id": "uuid-string",
        "status": "accepted|processing|completed|failed",
        "progress": 0-100,
        "stage": "stage-name",
        "message": "human-readable message",
        "result": {
            "r2_score": 0.882,
            "mae": 102.85,
            "rmse": 452.52
        },
        "error": null,
        "created_at": "2026-04-01T10:30:00Z",
        "updated_at": "2026-04-01T10:31:15Z"
    }
}
```

### Future: Redis Schema

```
Key: "training_job:{job_id}"
Value: JSON (same structure as above)
TTL: 86400 (24 hours)
```

### Future: PostgreSQL Schema

```sql
CREATE TABLE training_jobs (
    id UUID PRIMARY KEY,
    status VARCHAR(20),
    progress INT,
    stage VARCHAR(50),
    message TEXT,
    result JSONB,
    error TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_status ON training_jobs(status);
CREATE INDEX idx_created_at ON training_jobs(created_at DESC);
```

---

## Performance Considerations

### Optimization Strategies

#### 1. Job Cleanup
```python
# Remove completed jobs older than 24 hours
def cleanup_old_jobs(retention_hours=24):
    with TRAINING_JOBS_LOCK:
        cutoff_time = datetime.utcnow() - timedelta(hours=retention_hours)
        jobs_to_delete = []
        
        for job_id, job in TRAINING_JOBS.items():
            if job['status'] in ['completed', 'failed']:
                if datetime.fromisoformat(job['updated_at']) < cutoff_time:
                    jobs_to_delete.append(job_id)
        
        for job_id in jobs_to_delete:
            del TRAINING_JOBS[job_id]
        
        return len(jobs_to_delete)
```

#### 2. Concurrent Training Limit
```python
# Prevent resource exhaustion from too many concurrent trains
class TrainingJobQueue:
    def __init__(self, max_concurrent=1):
        self.max_concurrent = max_concurrent
        self.semaphore = threading.Semaphore(max_concurrent)
    
    def submit(self, job_id, func):
        def run_with_limit():
            with self.semaphore:
                func()
        
        thread = threading.Thread(target=run_with_limit, daemon=True)
        thread.start()
```

#### 3. Polling Interval Strategies
```javascript
// Adaptive polling: start fast, slow down as time progresses
function adaptivePolling(jobId, startInterval=1000, maxInterval=5000) {
    let interval = startInterval;
    
    async function poll() {
        const resp = await fetch(`/upload/status/${jobId}/`);
        const payload = await resp.json();
        
        // Update UI...
        updateProgress(payload);
        
        if (payload.status === 'completed' || payload.status === 'failed') {
            return;  // Stop polling
        }
        
        // Increase interval over time (exponential backoff-like)
        interval = Math.min(interval * 1.1, maxInterval);
        
        setTimeout(poll, interval);
    }
    
    poll();
}

// Usage: adaptivePolling(jobId)
```

### Benchmarks

| Scenario | Time | Memory |
|----------|------|--------|
| Training 1M rows | ~90s | ~2GB |
| Drift check 1M rows | ~5s | ~500MB |
| Status poll | <50ms | <1MB |
| 100 concurrent polls | ~500ms | ~10MB |

### Resource Monitoring

```python
import psutil

def monitor_resources():
    process = psutil.Process()
    
    return {
        'cpu_percent': process.cpu_percent(),
        'memory_mb': process.memory_info().rss / 1024 / 1024,
        'threads': threading.active_count(),
        'open_files': len(process.open_files())
    }
```

---

## Troubleshooting

### Job Status Stuck at "processing"
- **Cause**: Background thread crashed silently
- **Fix**: Add logging to `_run_training_job()`:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  
  try:
      # training logic
  except Exception as e:
      logger.error(f"Training job {job_id} failed: {e}")
      _update_job(job_id, status='failed', error=str(e))
  ```

### Drift Detection Returns 100% Drifted
- **Cause**: Comparing very different time periods (e.g., 2013 vs 2017)
- **Fix**: Lower threshold or use relative drift tolerance:
  ```python
  # Instead of absolute z-score, use relative change
  relative_change = abs((new_mean - ref_mean) / ref_mean)
  is_drifted = relative_change > 0.3  # 30% change
  ```

### Memory Leaks from Temp Files
- **Cause**: Temp files not deleted after training
- **Fix**: Use context manager:
  ```python
  import tempfile
  
  with tempfile.NamedTemporaryFile(delete=True) as tmp:
      # Save file to tmp.name
      # Automatically deleted on context exit
      pass
  ```

---

## References

- [scikit-learn Pipeline docs](https://scikit-learn.org/stable/modules/compose.html#pipeline)
- [Pandas 2.2.x Migration Guide](https://pandas.pydata.org/docs/user_guide/integer_na.html)
- [FastAPI Async TasksInstrumentation](https://fastapi.tiangolo.com/advanced/background-tasks/)
- [Threading Safety in Python](https://docs.python.org/3/library/threading.html)

---

**Last Updated**: April 1, 2026  
**Author**: Allen (OmniFlow Development Team)
