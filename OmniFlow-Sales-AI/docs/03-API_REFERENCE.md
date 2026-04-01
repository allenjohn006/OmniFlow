# 📡 API Reference

## Base URL

- **Development**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs` (Swagger UI)
- **ReDoc**: `http://localhost:8000/redoc`

---

## Authentication

Currently: None (development mode)

**Production**: Implement JWT bearer tokens in future versions

---

## Response Format

All responses are JSON:

```json
{
  "status": "success|error",
  "data": { ... },
  "error": "error message (if failed)"
}
```

---

## Endpoints

### Health Check

Check if API is running.

```http
GET /health
```

**Response** `200 OK`:
```json
{
  "status": "ok",
  "timestamp": "2026-04-01T10:30:00Z"
}
```

---

## Training Endpoints

### Start Training Job

Upload CSV and start async model training.

```http
POST /train/start
```

**Parameters**:
- `file` (multipart/form-data, required): CSV file with sales data
- `target_col` (form data, optional): Target column name (default: `"sales"`)

**Request**:
```bash
curl -X POST \
  -F "file=@data/raw/combined.csv" \
  -F "target_col=sales" \
  http://localhost:8000/train/start
```

**Response** `200 OK`:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "progress": 0,
  "stage": "preprocessing",
  "message": "Job queued for processing",
  "created_at": "2026-04-01T10:30:00Z"
}
```

**Response** `400 Bad Request`:
```json
{
  "detail": "No file provided"
}
```

---

### Get Training Status

Poll progress of async training job.

```http
GET /train/status/{job_id}
```

**Path Parameters**:
- `job_id` (string, required): UUID returned from `/train/start`

**Request**:
```bash
curl -X GET http://localhost:8000/train/status/550e8400-e29b-41d4-a716-446655440000
```

**Response** `200 OK` (Running):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": 65,
  "stage": "training",
  "message": "Training XGBoost model (iteration 200/300)",
  "updated_at": "2026-04-01T10:35:00Z",
  "result": null,
  "error": null
}
```

**Response** `200 OK` (Completed):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "stage": "done",
  "message": "Training completed successfully",
  "result": {
    "r2_score": 0.8805,
    "mae": 103.52,
    "rmse": 145.23,
    "train_samples": 2500000,
    "test_samples": 500000,
    "features_count": 15,
    "model_file": "models/champion.joblib",
    "metrics_file": "models/champion_metrics.json",
    "training_time_seconds": 285.50
  },
  "updated_at": "2026-04-01T10:40:30Z"
}
```

**Response** `200 OK` (Failed):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "progress": 45,
  "stage": "training",
  "message": "OOM error during model training",
  "error": "MemoryError: Unable to allocate 2GB for training matrix",
  "updated_at": "2026-04-01T10:35:00Z"
}
```

**Response** `404 Not Found`:
```json
{
  "detail": "Job not found"
}
```

---

## Drift Detection Endpoints

### Start Drift Detection

Upload new data and detect drift + auto-retrain if needed.

```http
POST /drift/start
```

**Parameters**:
- `file` (multipart/form-data, required): CSV with new data to analyze
- `target_col` (form data, optional): Target column (default: `"sales"`)

**Request**:
```bash
curl -X POST \
  -F "file=@data/raw/drift_test.csv" \
  -F "target_col=sales" \
  http://localhost:8000/drift/start
```

**Response** `200 OK`:
```json
{
  "job_id": "660e8400-e29b-41d4-a716-446655440111",
  "status": "queued",
  "progress": 0,
  "stage": "preprocessing",
  "message": "Drift detection job queued",
  "created_at": "2026-04-01T11:00:00Z"
}
```

---

### Get Drift Status

```http
GET /drift/status/{job_id}
```

**Response** `200 OK` (Drift Detected, No Retrain):
```json
{
  "job_id": "660e8400-e29b-41d4-a716-446655440111",
  "status": "completed",
  "progress": 100,
  "stage": "done",
  "result": {
    "drift_detected": true,
    "drift_ratio": 0.7333,
    "drifted_features": 11,
    "total_features": 15,
    "champion_metrics_baseline": {
      "r2_score": 0.8820,
      "mae": 102.85,
      "rmse": 144.50
    },
    "new_data_metrics": {
      "r2_score": 0.8805,
      "mae": 103.52,
      "rmse": 145.23
    },
    "performance_degraded": false,
    "recommendation": "Champion Retained",
    "decision_reason": "Drift detected, but champion metrics stayed within guardrails (R² drop 0.17% < 5% threshold)",
    "retrain_triggered": false,
    "new_samples": 500000
  },
  "updated_at": "2026-04-01T11:05:00Z"
}
```

**Response** `200 OK` (Drift Detected, Retrain Triggered):
```json
{
  "job_id": "660e8400-e29b-41d4-a716-446655440111",
  "status": "completed",
  "progress": 100,
  "stage": "done",
  "result": {
    "drift_detected": true,
    "drift_ratio": 0.8667,
    "drifted_features": 13,
    "total_features": 15,
    "champion_metrics_baseline": {
      "r2_score": 0.8820,
      "mae": 102.85,
      "rmse": 144.50
    },
    "new_data_metrics": {
      "r2_score": 0.7850,
      "mae": 125.30,
      "rmse": 165.40
    },
    "performance_degraded": true,
    "recommendation": "Champion Retained - Challenger Not Promoted",
    "decision_reason": "Drift detected with performance degradation (R² drop 11.0% > 5%). Challenger trained but champion remained better.",
    "retrain_triggered": true,
    "challenger_metrics": {
      "r2_score": 0.7720,
      "mae": 128.45,
      "rmse": 170.20
    },
    "champion_retained": true,
    "new_samples": 500000,
    "retraining_time_seconds": 290.5
  },
  "updated_at": "2026-04-01T11:20:30Z"
}
```

---

## Prediction Endpoints

### Single Prediction

Make a real-time prediction for a single item.

```http
POST /predict
```

**Parameters** (JSON body):
```json
{
  "store_nbr": 1,
  "family": "AUTOMOTIVE",
  "onpromotion": 0,
  "cluster": 5,
  "dcoilwtico": 98.5,
  "lag_7": 2150,
  "year": 2016,
  "month": 1,
  "dayofweek": 3,
  "is_weekend": 0,
  "is_salary_day": 0,
  "city": "Quito",
  "state": "Pichincha",
  "type": "A",
  "holiday_type": "None"
}
```

**Request**:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "store_nbr": 1,
    "family": "AUTOMOTIVE",
    "onpromotion": 0,
    "cluster": 5,
    "dcoilwtico": 98.5,
    "lag_7": 2150,
    "year": 2016,
    "month": 3,
    "dayofweek": 3,
    "is_weekend": 0,
    "is_salary_day": 0,
    "city": "Quito",
    "state": "Pichincha",
    "type": "A",
    "holiday_type": "None"
  }' \
  http://localhost:8000/predict
```

**Response** `200 OK`:
```json
{
  "prediction": 1852.45,
  "confidence_interval": {
    "lower": 1750.23,
    "upper": 1954.67
  },
  "model_info": {
    "version": "champion",
    "r2_score": 0.8805,
    "mae": 103.52
  }
}
```

**Response** `400 Bad Request`:
```json
{
  "detail": "Missing required field: store_nbr"
}
```

---

## Data Format Reference

### Input CSV Format (Training & Drift)

```csv
date,store_nbr,family,sales,onpromotion,cluster,dcoilwtico,lag_7,year,month,dayofweek,is_weekend,is_salary_day,city,state,type,holiday_type
2015-01-01,1,AUTOMOTIVE,0.00,0,14,92.89,1788,2015,1,3,0,0,Quito,Pichincha,A,New Year
2015-01-02,1,AUTOMOTIVE,1500.00,1,14,92.89,1800,2015,1,4,0,0,Quito,Pichincha,A,None
```

**Required Columns**:
- `date` - Date in YYYY-MM-DD format (extracted for features, not used as model input)
- `store_nbr` - Store identifier (numeric)
- `family` - Product family (categorical)
- `sales` - Target value (numeric, for training only)
- `onpromotion` - Is item on promotion (0 or 1)
- `cluster` - Store cluster assignment
- `dcoilwtico` - Oil price
- `lag_7` - 7-day lag of sales (or similar)
- `year`, `month`, `dayofweek` - Date components (numeric)
- `is_weekend` - Binary indicator
- `is_salary_day` - Binary indicator
- `city`, `state`, `type` - Categorical features
- `holiday_type` - Holiday classification or "None"

---

## Status Codes

| Code | Meaning |
|------|---------|
| `200` | Success |
| `400` | Bad request (missing/invalid parameters) |
| `404` | Job not found |
| `422` | Validation error (invalid data) |
| `500` | Server error |
| `503` | Service temporarily unavailable |

---

## Rate Limiting

Currently: None (development mode)

**Production limits** (to be implemented):
- `/predict`: 100 req/sec per IP
- `/train/start`: 1 job per user (to prevent spam)
- `/drift/start`: 1 job per user at a time

---

## Error Handling

**Example error response**:
```json
{
  "detail": "Feature schema mismatch. Expected columns: [store_nbr, family, ...], got: [store_id, product_family, ...]"
}
```

---

## Async Job Polling Pattern

All long-running operations follow this pattern:

1. **Submit Job**:
   ```bash
   POST /train/start → {job_id, status: queued}
   ```

2. **Poll Status** (every 2 seconds):
   ```bash
   GET /train/status/{job_id} → {status, progress, stage}
   ```

3. **Wait for Completion**:
   ```
   Loop until status in [completed, failed]
   ```

4. **Get Results**:
   ```json
   {status: completed, result: {...metrics...}}
   ```

---

## Example: Complete Training Flow

```python
import requests
import time

BASE_URL = "http://localhost:8000"

# 1. Start training
response = requests.post(
    f"{BASE_URL}/train/start",
    files={"file": open("data/raw/combined.csv", "rb")},
    data={"target_col": "sales"}
)
job_id = response.json()["job_id"]
print(f"Training started: {job_id}")

# 2. Poll until complete
while True:
    status = requests.get(f"{BASE_URL}/train/status/{job_id}").json()
    print(f"Progress: {status['progress']}% - {status['stage']}")
    
    if status["status"] in ["completed", "failed"]:
        break
    
    time.sleep(2)

# 3. Show results
if status["status"] == "completed":
    result = status["result"]
    print(f"✅ Training complete!")
    print(f"   R² Score: {result['r2_score']:.4f}")
    print(f"   MAE: {result['mae']:.2f}")
    print(f"   RMSE: {result['rmse']:.2f}")
```

