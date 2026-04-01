# 🚀 OmniFlow Sales AI

A production-grade MLOps platform for store sales forecasting with **async training**, **drift detection**, and **real-time progress tracking**.

## 📋 Overview

OmniFlow sales AI is a full-stack machine learning system that:
- **Trains XGBoost models** on large e-commerce sales datasets (3M+ rows)
- **Detects data drift** to monitor model performance degradation
- **Provides real-time progress** during long-running training jobs
- **Ensures numerical stability** with explicit dtype management for pandas 2.2+
- **Scales horizontally** with async background job processing

**Built with**: FastAPI (backend) + Django (frontend) + XGBoost (ML) + scikit-learn (preprocessing)

---

## 🎯 Key Features

### ✅ Async Training Architecture
- **Non-blocking job submission**: Training runs in background thread
- **Live progress tracking**: 8-phase progress callbacks during model fit
- **Real-time UI updates**: Browser polls every 2s for live stage/percent/message
- **Timeout elimination**: 120s+ training jobs no longer block HTTP requests

### ✅ Drift Detection
- **Statistical comparison**: Detects feature distribution shifts over time
- **Per-feature drift metrics**: Identifies which features changed most
- **Retrain triggers**: Signals when model needs updating based on data drift
- **Reference baselines**: Compares new data against training data statistics

### ✅ Data Quality Assurance
- **Explicit dtype casting**: Forces float64 on all numerical features (pandas 2.2+ safety)
- **Schema validation**: Ensures feature columns match expected order before model fit
- **Datetime handling**: Properly separates date columns from model input
- **Missing value management**: Fills with sensible defaults (0 for numeric, "None" for categorical)

### ✅ Production-Ready ML Pipeline
- **Unified scikit-learn Pipeline**: ColumnTransformer + OneHotEncoder + XGBoostRegressor
- **Temporal data split**: Train (< 2016-01-01) vs Test (>= 2016-01-01)
- **Feature engineering**: Date extractions, lag features, binary indicators, merged auxiliaries
- **Model persistence**: Single joblib file for reproducible inference

---

## 🏗️ Architecture

### Backend Stack (FastAPI)
```
api/main.py
├── GET  /health              → Health check
├── POST /train/start          → Start async training job (returns job_id)
├── GET  /train/status/{job_id}→ Poll training progress
├── POST /drift               → Check for data drift
└── POST /predict             → Single prediction endpoint
```

### Frontend Stack (Django)
```
django_app/omniapp/
├── views.py
│   ├── upload()              → Upload CSV, trigger async training
│   ├── upload_status()       → Proxy status polls to FastAPI
│   ├── train_result()        → Display final metrics & model info
│   ├── drift_result()        → Display drift analysis report
│   └── predict()             → Single prediction interface
├── templates/
│   ├── index.html            → Dashboard home
│   ├── upload.html           → Training interface + live progress bar
│   ├── train_result.html     → Success page with metrics cards
│   ├── drift_result.html     → Drift analysis report
│   └── predict.html          → Prediction interface
└── urls.py                   → Route definitions
```

### ML Pipeline (src/)
```
src/
├── training.py         → Model training with progress callbacks
├── preprocessing.py    → Feature engineering & dtype management
├── drift.py           → Statistical drift detection
├── retrain.py         → Retraining logic (challenger models)
├── ingestion.py       → Data loading & validation
└── utils.py           → Helper functions
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pandas 2.2.2
- scikit-learn 1.6.0
- XGBoost 2.1.1
- FastAPI 0.111.0
- Django 5.0.4

### Installation

1. **Clone repository**
```bash
git clone <repo>
cd OmniFlow-Sales-AI
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare data** (optional - creates drift test set)
```bash
python create_drift_test_data.py
```

### Running the Application

**Start both FastAPI + Django servers:**
```bash
python dev.py run-all
```

This launches:
- 🟦 **FastAPI**: http://localhost:8000 (ML backend)
- 🟨 **Django**: http://localhost:8080 (Web frontend)

### Usage Workflow

#### 1. **Train New Model**
```
Dashboard → Train Tab → Upload CSV
├── Select file (e.g., data/raw/train.csv)
├── Enter target column (e.g., "sales")
└── Click "🚀 Train Model"
```

**What happens:**
- Form submission → FastAPI `/train/start` (async)
- Job ID returned immediately (no 120s timeout!)
- Browser polls `/upload/status/{job_id}` every 2s
- Live progress bar shows stage + percent + message
- On 100% completion → auto-redirect to results page

**Example progress stages:**
```
5% - Loading data...
20% - Splitting train/test...
35% - Building pipeline...
55% - Training XGBoost...
82% - Evaluating metrics...
92% - Saving model...
100% - Complete!
```

#### 2. **Check for Drift**
```
Dashboard → Drift & Retrain Tab → Upload CSV
├── Select newer data file (e.g., data/raw/drift_test.csv)
├── Enter target column
└── Click "🔍 Check for Drift"
```

**Output:**
- **Drift ratio**: % of features that drifted
- **Per-feature comparison**: Reference mean vs new mean
- **Deviation scores**: How much each feature changed
- **Status per feature**: ✓ Stable or ⚠️ Drifted
- **Recommendation**: Should you retrain? (auto-triggered if drift > threshold)

#### 3. **Make Predictions**
```
Dashboard → Predict Tab → Enter feature values
├── store_nbr, date, family, etc.
└── Click "🎯 Predict"
```

Returns: **Predicted sales value** (scaled to original units)

---

## 📊 Data Format

### Input CSV Requirements
Required columns:
- `date` (format: YYYY-MM-DD)
- `store_nbr` (integer)
- `family` (string - product category)
- `sales` (float - target variable)

Optional columns (auto-merged if available):
- `cluster` (store cluster ID)
- `oil` (dcoilwrico prices)
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
- **Training Data**: Kaggle Store Sales (2013-2015, temporal split)
- **Metrics**:
  - R² Score: **0.882** (88.2% variance explained)
  - MAE: **102.85** (mean absolute error)
  - RMSE: **452.52** (root mean squared error)

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

```
OmniFlow-Sales-AI/
├── api/                          # FastAPI backend
│   ├── main.py                  # Core endpoints
│   └── routes/                  # Modular route definitions
├── src/                          # ML pipeline modules
│   ├── training.py              # Model training + progress
│   ├── preprocessing.py         # Feature engineering
│   ├── drift.py                 # Drift detection
│   └── utils.py                 # Helpers
├── django_app/                   # Django frontend
│   ├── omniapp/
│   │   ├── views.py             # Request handlers
│   │   ├── urls.py              # URL routing
│   │   └── templates/           # HTML templates
│   └── manage.py                # Django CLI
├── data/
│   ├── raw/                     # Original CSVs
│   ├── processed/               # Cleaned/cached data
│   └── reference_stats.json     # Baseline distributions
├── models/
│   ├── champion.joblib          # Production model
│   └── champion_metrics.json    # Last training metrics
├── tests/                        # Unit & integration tests
├── dev.py                        # Local dev server launcher
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

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

## 📝 License

MIT License - See LICENSE file for details

---

## 👥 Authors

- **Development**: Allen (OmniFlow Team)
- **ML Architecture**: XGBoost + scikit-learn
- **Web Stack**: FastAPI + Django

---

## 📞 Support

For issues, questions, or feature requests:
1. Check [Troubleshooting](#-troubleshooting) section
2. Review [CHANGELOG.md](CHANGELOG.md) for recent changes
3. Open an issue on GitHub with detailed reproduction steps

---

**Last Updated**: April 1, 2026  
**Current Branch**: TimeSeries  
**Status**: ✅ Production Ready
