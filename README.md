# 🚀 OmniFlow Sales AI

A production-grade MLOps platform for store sales forecasting with **async training**, **intelligent drift detection**, **auto-retraining**, and **real-time progress tracking**.

> **Latest Updates**: ✨ Full async drift pipeline with auto-retrain decision engine • 📊 Live progress tracking with percentage • 🛡️ JSON serialization safety for numpy/pandas types • 📚 Comprehensive documentation suite

---

## 📋 Overview

OmniFlow Sales AI is an enterprise-ready machine learning system that:

- **🏋️ Trains XGBoost models** on large e-commerce datasets (3M+ rows, <5 min training)
- **📈 Detects data drift** using statistical hypothesis testing (KS test, Chi-square)
- **🔄 Auto-triggers retraining** with intelligent decision logic (5% R², 10% MAE/RMSE guardrails)
- **⏱️ Provides real-time progress** during long-running jobs (8-phase async pipeline)
- **🛡️ Ensures numerical stability** with explicit dtype management for pandas 2.2+
- **🌐 Scales horizontally** with background job processing and async REST API

**Tech Stack**: 
- **Backend**: FastAPI (async REST) + FastAPI-Background Jobs (thread pool)
- **Frontend**: Django (server-rendered templates) + AJAX polling
- **ML Engine**: XGBoost (gradient boosting) + scikit-learn (preprocessing)
- **Data**: pandas 2.2+ (explicit dtype casting), scipy (statistical tests)

---

## 🎯 Key Features

### ✅ Intelligent Auto-Retrain Engine
- **Guardrail-based decisions**: R² drop >5% OR MAE/RMSE increase >10% → trigger retrain
- **Statistical drift ≠ Performance** drift: Detects both, acts only when needed
- **Challenger model evaluation**: New model tested against champion before promotion
- **Decision explanation**: Shows WHY retrain was triggered (or retained)
- **Example**: Drift detected in 73% of features, but R² dropped only 0.17% → Champion retained

### ✅ Async Training Architecture
- **Non-blocking API**: Training via background threads (no 120s HTTP timeout)
- **8-phase callbacks**: Loading → Preprocessing → Encoding → Training → Validation → Metrics → Saving → Done
- **Live progress UI**: Real-time percentage, stage name, and descriptive message
- **Browser polling**: 2-second intervals for responsive UX without server push

### ✅ Statistical Drift Detection
- **Multiple hypothesis tests**: Kolmogorov-Smirnov (numerical) + Chi-square (categorical)
- **Per-feature metrics**: Identifies exact features that changed + magnitude
- **Reference baseline**: Saved from training data, compared against new data
- **Drift ratio**: Aggregated % of features drifted across dataset
- **Example output**: "11 of 15 features drifted (73%) - Oil prices shifted, holidays changed"

### ✅ Production-Ready ML Pipeline
- **Scikit-learn Pipeline**: ColumnTransformer (encoding) → XGBoost (model)
- **Feature engineering**: Date components, lag features, binary indicators, merged auxiliaries
- **Temporal split**: Train pre-2016, test post-2016 (time-series respecting)
- **15 engineered features**: 5 categorical (family, city, state, type, holiday) + 10 numerical

### ✅ Data Quality Assurance
- **Explicit dtype casting**: `float64` on all numerics (pandas 2.2+ safety net)
- **Schema validation**: Exact feature match before model fit
- **Missing value handling**: Mean imputation (numeric), mode imputation (categorical)
- **Invalid value detection**: Alerts on out-of-range features

### ✅ Model Metrics & Interpretability
- **Comprehensive metrics**: R² (variance explained), MAE (avg error), RMSE (std error)
- **Baseline comparison**: Shows champion metrics vs new data performance
- **Decision reasoning**: "Why was champion retained/updated?" explained in plain English
- **Model provenance**: Training samples, test samples, training time tracked

---

## 🏗️ System Architecture

### End-to-End Data Flow

```
┌─── TRAINING FLOW ───┐
│                     │
│  1. User uploads    │
│     CSV file        │
│          ↓          │
│  2. Django form     │
│     → FastAPI       │
│  /train/start       │
│          ↓          │
│  3. Background      │
│     job spawned     │
│          ↓          │
│  4. 8-phase         │
│     pipeline        │
│     (5 min for 3M)  │
│          ↓          │
│  5. Metrics calc    │
│     (R²,MAE,RMSE)   │
│          ↓          │
│  6. Model saved    │
│     + stats saved   │
│          ↓          │
│  7. Browser polls    │
│     every 2s         │
│          ↓          │
│  8. Results page    │
│     auto-redirect   │
│                     │
└─────────────────────┘

┌── DRIFT + AUTO-RETRAIN FLOW ──┐
│                               │
│  1. User uploads              │
│     new data                  │
│          ↓                    │
│  2. Detect drift              │
│     (KS + Chi²)               │
│          ↓                    │
│  3. Evaluate champion         │
│     on new data               │
│          ↓                    │
│  4. Compare metrics           │
│     (baseline vs new)         │
│          ↓                    │
│  5. Decision logic:           │
│     ├─ R² drop >5%? → train   │
│     ├─ MAE up >10%? → train   │
│     ├─ Drift >75%? → train    │
│     └─ else → retain          │
│          ↓                    │
│  6. If retrain:               │
│     Train challenger,         │
│     compare, promote if best  │
│          ↓                    │
│  7. Results page              │
│     with recommendation       │
│                               │
└───────────────────────────────┘
```

### API Endpoints

**FastAPI Backend** (port 8000):
```
GET  /health                      → {status: "ok"}
POST /train/start                 → {job_id, status, progress}
GET  /train/status/{job_id}       → {status, progress, stage, result}
POST /drift/start                 → {job_id, status}
GET  /drift/status/{job_id}       → {status, result: {drift_ratio, recommendation}}
POST /predict                     → {prediction, confidence_interval}
```

**Django Frontend** (port 8080):
```
GET  /                            → Dashboard
GET  /upload/                     → Training interface
GET  /predict/                    → Prediction interface
GET  /drift/                      → Drift analysis interface
GET  /drift_result/               → Drift analysis results
```

### Directory Structure

```
OmniFlow-Sales-AI/
├── 📁 api/                     # FastAPI backend
│   └── main.py                # Core endpoints + job orchestration
├── 📁 django_app/             # Django frontend
│   ├── omniapp/
│   │   ├── views.py           # HTTP handlers
│   │   ├── urls.py            # Routing
│   │   ├── static/            # CSS, JS, images
│   │   └── templates/         # HTML pages
│   └── manage.py              # Django CLI
├── 📁 src/                    # ML pipeline (core logic)
│   ├── training.py            # Model training + metrics
│   ├── preprocessing.py       # Feature engineering
│   ├── drift.py               # Drift detection (KS + Chi²)
│   ├── retrain.py             # Auto-retrain decision logic
│   ├── ingestion.py           # Data loading
│   ├── inference.py           # Predictions
│   └── utils.py               # Helpers
├── 📁 data/                   # Data directory (git-ignored)
│   ├── raw/                   # ✅ Source CSV files (required)
│   │   ├── train.csv          # Main timeseries data (3M+ rows)
│   │   ├── stores.csv         # Store metadata
│   │   ├── oil.csv            # Oil price timeseries
│   │   ├── holidays_events.csv # Holiday calendar
│   │   └── drift_test.csv     # Test data for drift examples
│   └── processed/             # *Note: Engineered features generated dynamically*
├── 📁 models/                 # Model artifacts (git-ignored)
│   ├── champion.joblib        # Trained XGBoost model
│   ├── champion_metrics.json  # Baseline metrics (R², MAE, RMSE)
│   ├── reference_stats.json   # Feature stats for drift comparison
│   └── feature_columns.json   # Feature schema definition
├── 📁 docs/                   # **NEW** Documentation suite
│   ├── 01-ARCHITECTURE.md     # Design & components
│   ├── 02-GETTING_STARTED.md  # Setup & quick start
│   ├── 03-API_REFERENCE.md    # Endpoint docs + examples
│   ├── 04-ML_PIPELINE.md      # ML details + algorithms
│   ├── 05-CONTRIBUTING.md     # Code standards + PR process
│   └── 06-DEPLOYMENT.md       # Production setup
├── dev.py                     # Development launcher (CLI)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── CHANGELOG.md               # Release notes
```

---

## 🚀 Quick Start

### 1. Installation & Data Setup

1. **Clone repository**:
   ```bash
   git clone https://github.com/your-org/OmniFlow-Sales-AI.git
   cd OmniFlow-Sales-AI
   ```

2. **Create & activate virtual environment**:
   ```bash
   # On Windows (PowerShell):
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # On macOS/Linux:
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Dataset**:
   Place the Kaggle Store Sales dataset files (`train.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`) in a directory named `data/` at the **parent** level of the project repository (i.e., `../data/`), or in the project fallback directory `data/raw/`.

---

### 2. Start Servers

You can launch both servers simultaneously using the development utility, or start them separately in different terminal windows.

#### Option A: Run both servers together (Recommended)
```bash
python dev.py run-all
```
This utility automatically runs FastAPI on port `8000` and Django on port `8080`.

#### Option B: Run servers individually
* **FastAPI Backend (Terminal 1)**:
  ```bash
  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
  ```
* **Django Frontend (Terminal 2)**:
  ```bash
  cd django_app
  python manage.py runserver 0.0.0.0:8080
  ```

Once started, access the interfaces:
- 🌐 **Web Dashboard**: [http://localhost:8080](http://localhost:8080)
- 📖 **API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 3. Use the System

**Train**: [http://localhost:8080/upload/](http://localhost:8080/upload/)
- Upload `train.csv` (target column: `sales`) → See 8-phase live progress → View metrics (R², MAE, RMSE)

**Drift**: [http://localhost:8080/drift/](http://localhost:8080/drift/)
- Upload `drift_test.csv` (target column: `sales`) → Auto-detect drift → See auto-retrain decision (with reason)

**Predict**: [http://localhost:8080/predict/](http://localhost:8080/predict/)
- Enter features → Get real-time sales prediction

---

## 📊 Example Workflows

### Workflow 1: Train a Model (5 minutes)

```bash
1. Go to http://localhost:8080/upload/
2. Choose file: data/raw/train.csv (Store Sales Ecuador timeseries)
3. Click "Upload" and watch live progress (8 phases)
4. See training results: R², MAE, RMSE metrics
```

**Expected Output**:
```
Training Progress:
✓ Data Loading (10%)
✓ Feature Preprocessing (30%)
✓ Model Training (55%)
✓ Predictions & Metrics (75%)
✓ Model Saving (88%)
✓ Complete (100%)

Results:
R² Score: 0.8234
MAE: 2,145.67
RMSE: 3,456.78
```

---

### Workflow 2: Detect Drift & Auto-Retrain

```bash
1. Go to http://localhost:8080/drift/
2. Choose file: data/raw/drift_test.csv (2024 Q1 data, post-2016)
3. Target: "sales"
4. Click "🚀 Train Model"
5. Watch live progress: Loading (10%) → Preprocessing (20%) → ... → Done (100%)
6. View results: R²=0.8805, MAE=103.52, RMSE=145.23
```

### Workflow 2: Check for Drift + Auto-Retrain

```bash
1. Go to http://localhost:8080/drift/
2. Select: data/raw/drift_test.csv (2024 Q1 data)
3. Target: "sales"
4. Click "🔍 Check for Drift"
5. Results show:
   ├─ Drift detected: 73% of features (11/15)
   ├─ Champion R²: 0.8820 (baseline) → 0.8805 (new) = -0.17% drop
   ├─ Decision: "Champion Retained"
   ├─ Reason: "Drop < 5% threshold, metrics stable"
   └─ Recommendation: "Continue monitoring"
```

### Workflow 3: Make Predictions

```bash
1. Go to http://localhost:8080/predict/
2. Enter: store_nbr=1, family="PRODUCE", city="Quito", ...
3. Click "🎯 Predict"
4. Result: $1,852.45 (predicted daily sales)
```

---

## 🔧 API Usage Examples

### Python Client

```python
import requests
import time

BASE_URL = "http://localhost:8000"

# 1. Start training
response = requests.post(
    f"{BASE_URL}/train/start",
    files={"file": open("data/raw/train.csv", "rb")},
    data={"target_col": "sales"}
)
job_id = response.json()["job_id"]
print(f"✅ Training started: {job_id}")

# 2. Poll until complete
while True:
    status = requests.get(f"{BASE_URL}/train/status/{job_id}").json()
    print(f"Progress: {status['progress']}% - {status['stage']}")
    
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(2)

# 3. Show results
if status["status"] == "completed":
    metrics = status["result"]
    print(f"✅ Training complete!")
    print(f"   R² Score: {metrics['r2_score']:.4f}")
    print(f"   MAE: {metrics['mae']:.2f}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Interactive API docs
open http://localhost:8000/docs

# Prediction example
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"store_nbr": 1, "family": "PRODUCE", ...}'
```

---

## 📚 Documentation

Complete documentation available in `/docs`:

| File | Purpose |
|------|---------|
| [01-ARCHITECTURE.md](docs/01-ARCHITECTURE.md) | System design, components, data flow |
| [02-GETTING_STARTED.md](docs/02-GETTING_STARTED.md) | Setup, troubleshooting, tips |
| [03-API_REFERENCE.md](docs/03-API_REFERENCE.md) | All endpoints, request/response formats |
| [04-ML_PIPELINE.md](docs/04-ML_PIPELINE.md) | ML algorithms, metrics, drift logic |
| [05-CONTRIBUTING.md](docs/05-CONTRIBUTING.md) | Code standards, testing, PRs |
| [06-DEPLOYMENT.md](docs/06-DEPLOYMENT.md) | Docker, AWS, GCP, production setup |

---

## 💡 Key Concepts

### Auto-Retrain Decision Logic

**Problem**: When should a model be retrained?
- Always = waste of compute, model churn
- Never = miss performance degradation
- Smart = only when needed ✓

**Solution**: Guardrail-based thresholds:

```
IF   R² drops > 5%              → RETRAIN
  OR MAE increases > 10%        → RETRAIN  
  OR RMSE increases > 10%       → RETRAIN
  OR drift_ratio > 75%          → RETRAIN
ELSE → RETAIN champion
```

**Real example**: 
- 73% features drifted (drift_ratio > 75% threshold)
- BUT R² only dropped -0.17% (well below 5% threshold)
- **Decision**: Champion Retained (better safe than sorry!)

### Drift ≠ Performance

- **Data Drift**: Input feature distributions changed
- **Performance Drift**: Model predictions became inaccurate
- **Insight**: Drift can happen WITHOUT harming performance, and vice versa
- **System Design**: Detect drift + evaluate performance + decide intelligently

### Async Job Architecture

```
User uploads file
       ↓
POST /train/start → immediate response {job_id}
       ↓
Background thread starts processing (no blocking!)
       ↓
Browser polls /train/status/{job_id} every 2s
       ↓
Progress updates: 0% → 50% → 100% (real-time)
       ↓
On 100%, browser auto-redirects to results
```

---

## 🎓 Learning Path

1. **Understand the flow**: Read [ARCHITECTURE.md](docs/01-ARCHITECTURE.md)
2. **Get it running**: Follow [GETTING_STARTED.md](docs/02-GETTING_STARTED.md)
3. **Explore the API**: Test examples in [API_REFERENCE.md](docs/03-API_REFERENCE.md)
4. **Learn the ML**: Deep dive into [ML_PIPELINE.md](docs/04-ML_PIPELINE.md)
5. **Deploy to production**: Follow [DEPLOYMENT.md](docs/06-DEPLOYMENT.md)
6. **Contribute**: See [CONTRIBUTING.md](docs/05-CONTRIBUTING.md)

---

## 📊 Performance Metrics

**Training Performance**:
| Dataset Size | Time | CPU |
|---|---|---|
| 500K rows | ~30s | Single core |
| 1.5M rows | ~90s | 4 core |
| 3M rows | ~280s (4.7min) | 8 core |

**Inference Performance**:
| Batch Size | Latency |
|---|---|
| Single | <50ms |
| 1000 | ~200ms |

**Model Accuracy**:
| Metric | Value |
|---|---|
| R² Score | 0.8820 |
| MAE | 102.85 |
| RMSE | 452.52 |
| Train samples | 1,945,944 |
| Test samples | 1,054,944 |

---

## 🛠️ Troubleshooting

**Port already in use?**
```bash
# Kill process on port 8000
lsof -ti :8000 | xargs kill -9    # macOS/Linux
# or Windows: Get-NetTCPConnection -LocalPort 8000 | Stop-Process -Force
```

**ModuleNotFoundError: No module named 'src'?**
```bash
# Ensure you're in the project root
cd OmniFlow-Sales-AI
python dev.py run-all
```

**See [GETTING_STARTED.md](docs/02-GETTING_STARTED.md) for more troubleshooting**

---

## 🚀 What's New (Recent Updates)

✨ **Latest Release** (Apr 2026):
- Added async drift detection with auto-retrain decision engine
- Live progress tracking UI with percentage display
- Guardrail-based retrain logic (5% R² threshold, 10% MAE/RMSE)
- Comprehensive documentation suite (6 guides)
- JSON serialization safety for numpy/pandas types
- Django + FastAPI integration for async job polling

📝 See [CHANGELOG.md](CHANGELOG.md) for version history

---

## 📄 License

MIT License - See LICENSE file

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](docs/05-CONTRIBUTING.md) for:
- Code standards (PEP 8, black, flake8)
- Testing requirements
- Git commit conventions
- PR process

---



---

## 📈 Roadmap

- [ ] Multi-model ensemble (XGBoost + LightGBM + CatBoost)
- [ ] SHAP values for model explainability
- [ ] Auto hyperparameter tuning (Bayesian optimization)
- [ ] GPU acceleration for XGBoost training
- [ ] Kubernetes deployment templates
- [ ] Redis job queue for horizontal scaling
- [ ] Advanced monitoring & alerting dashboard

---

**Made with ❤️ for production ML systems**
- `holiday_type` (holiday classification)

### Feature Engineering Pipeline
| Extracted/Computed | Type | Source |
|-------------------|------|--------|
| year, month, dayofweek | int | Date column |
| is_weekend | binary | dayofweek in [5,6] |
| is_salary_day | binary | day in [15,30] |
| lag_7 | float | Grouped shift (7 days) |
| dcoilwrico | float | Merged from oil.csv |
| holiday_type | categorical | Merged from holidays_events.csv |
| store cluster info | one-hot encoded | Merged from stores.csv |

### Example Raw Data
```csv
date,store_nbr,family,on_promotion,sales
2013-01-01,1,PRODUCE,0,100.5
2013-01-02,1,PRODUCE,0,120.3
...
```

---

## 🔧 Technical Details

### Dtype Handling (Pandas 2.2+ Safety)
**Problem**: Pandas 2.2.2 promotes nullable Int64 columns mixed with datetime64, causing ColumnTransformer failures.

**Solution** in `src/preprocessing.py`:
```python
# Explicitly cast ALL numerical features to float64
out[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
out[NUMERICAL_FEATURES] = out[NUMERICAL_FEATURES].fillna(0.0).astype('float64')

# Add schema validation before model input
validate_feature_frame(X_train)  # Asserts no datetime columns
```

### Async Job Architecture
In `api/main.py`:
```python
TRAINING_JOBS = {}  # {job_id: {status, progress, stage, message, result}}
TRAINING_JOBS_LOCK = threading.Lock()  # Thread-safe updates

@app.post("/train/start")
async def start_training(file: UploadFile):
    job_id = str(uuid.uuid4())
    # Immediate response
    threading.Thread(target=_run_training_job, args=(job_id,), daemon=True).start()
    return {"status": "accepted", "job_id": job_id, "poll_url": f"/train/status/{job_id}"}

@app.get("/train/status/{job_id}")
async def get_status(job_id: str):
    with TRAINING_JOBS_LOCK:
        return TRAINING_JOBS.get(job_id, {"status": "not_found"})
```

### Progress Callback System
In `src/training.py`:
```python
def train_model(train_file, test_file, target_col, progress_callback=None):
    if progress_callback:
        progress_callback(5, "Loading", "Loading data...")
    
    df = pd.read_csv(train_file)
    
    if progress_callback:
        progress_callback(35, "Pipeline", "Building feature transformer...")
    
    pipeline = Pipeline([...])
    
    if progress_callback:
        progress_callback(55, "Training", "Fitting XGBoost (300 trees)...")
    
    pipeline.fit(X_train, y_train)
    # Final callback at 100%
```

### Drift Detection Algorithm
In `src/drift.py`:
```python
# Calculate z-score for each feature
drift_scores[feature] = abs((new_mean - ref_mean) / ref_std)

# Flag as drifted if z-score > 2 (95% confidence)
is_drifted = drift_scores[feature] > DRIFT_THRESHOLD

# Overall drift ratio
drift_ratio = n_drifted_features / total_features
```

---

## 📈 Model Performance

**Current Best Model** (as of last training):
- **Algorithm**: XGBoost
- **Hyperparameters**: 300 trees, max_depth=8, learning_rate=0.05
- **Training Data**: Kaggle Store Sales (temporal split: train < 2016-01-01, test >= 2016-01-01)
- **Metrics**:
  - R² Score: **0.8820** (88.2% variance explained)
  - MAE: **102.8503** (mean absolute error)
  - RMSE: **452.5207** (root mean squared error)
  - Train Rows: **1,945,944**
  - Test Rows: **1,054,944**

---

## 🧪 Testing

### Unit Tests
```bash
pytest tests/ -v
```

### Integration Tests
```bash
# Test async training flow
python -m pytest tests/test_async_training.py

# Test drift detection
python -m pytest tests/test_drift_detection.py
```

### Manual Testing
1. Start servers: `python dev.py run-all`
2. Navigate to http://localhost:8080
3. Upload test file: `data/raw/drift_test.csv`
4. Monitor progress bar in real-time
5. Verify results page renders correctly

---

## 📋 Project Structure

For a full breakdown of the directory layout and file roles, please refer to the [Directory Structure](#directory-structure) section at the top of this document.

---

## 🐛 Troubleshooting

### "Module not found: src"
```bash
# Ensure you're in project root
cd OmniFlow-Sales-AI
export PYTHONPATH=.
```

### "Port 8000/8080 already in use"
```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9  # Linux/macOS
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process  # Windows
```

### Training times out after 120s
✅ **Already fixed!** Async architecture removes blocking request timeout.

### "Int64DType vs DateTime64DType error"
✅ **Already fixed!** Explicit `astype('float64')` prevents promotion issues.

### Drift detection shows all features drifted
This is **expected** if comparing very different time periods (e.g., 2013 vs 2017). Expected behavior:
- ✅ If drift ratio < 50%: Model is stable
- ⚠️ If drift ratio 50-75%: Consider retraining
- 🔴 If drift ratio > 75%: Should retrain immediately

---

## 🚀 Deployment

### Docker
```bash
# Build image
docker build -t omniflow-sales-ai .

# Run container
docker run -p 8000:8000 -p 8080:8080 omniflow-sales-ai
```

### Production Checklist
- [ ] Update `.env` with production database URL
- [ ] Set `DEBUG=False` in Django settings
- [ ] Configure CORS for frontend domain
- [ ] Set up persistent job storage (Redis/PostgreSQL)
- [ ] Enable request logging & monitoring
- [ ] Configure email alerts for drift triggers

---

## 📚 API Documentation

### FastAPI Interactive Docs
Available at: http://localhost:8000/docs (Swagger UI)

### Key Endpoints

**POST /train/start**
```json
Request:
{
  "file": "<CSV file>",
  "target_col": "sales"
}

Response:
{
  "status": "accepted",
  "job_id": "89841818-cfb9-4b64-bccf-5ab37d9820dd",
  "poll_url": "/train/status/89841818-cfb9-4b64-bccf-5ab37d9820dd"
}
```

**GET /train/status/{job_id}**
```json
Response (in-progress):
{
  "status": "processing",
  "progress": 55,
  "stage": "Training",
  "message": "Fitting XGBoost (300 trees)...",
  "created_at": "2026-04-01T10:30:00Z"
}

Response (completed):
{
  "status": "completed",
  "progress": 100,
  "stage": "done",
  "result": {
    "r2_score": 0.882,
    "mae": 102.85,
    "rmse": 452.52
  }
}
```

---

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and test
3. Commit: `git commit -m "Add feature description"`
4. Push: `git push origin feature/your-feature`
5. Create Pull Request

---


## 👥 Authors

- **Development**: Allen (OmniFlow Team)
- **ML Architecture**: XGBoost + scikit-learn
- **Web Stack**: FastAPI + Django

---

---

**Last Updated**: July 19, 2026  
**Current Branch**: TimeSeries  
**Status**: ✅ Production Ready
