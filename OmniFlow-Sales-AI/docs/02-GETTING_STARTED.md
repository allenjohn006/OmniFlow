# 🚀 Getting Started Guide

## Prerequisites

- **Python**: 3.9 or higher
- **OS**: Windows, macOS, or Linux
- **Memory**: 8GB RAM minimum (16GB recommended for training)
- **Disk**: 5GB free space for data and models

## Development Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/OmniFlow-Sales-AI.git
cd OmniFlow-Sales-AI
```

### 2. Create Virtual Environment

**On Windows**:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On macOS/Linux**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Installed Packages**:
- `fastapi>=0.111.0` - Web framework
- `uvicorn[standard]>=0.29.0` - ASGI server
- `django>=5.0.4` - Frontend framework
- `pandas>=2.2.2` - Data manipulation
- `scikit-learn>=1.6.0` - ML preprocessing & metrics
- `xgboost>=2.1.1` - Gradient boosting model
- `joblib>=1.4.0` - Model serialization
- `numpy>=2.2.0` - Numerical computing
- `scipy>=1.13.1` - Statistical tests (drift detection)
- `requests>=2.31.0` - HTTP client
- All others for visualization and utilities

### 4. Prepare Data

Your training data should be in `data/raw/combined.csv` with structure:
```
date,store_nbr,family,sales,onpromotion,cluster,dcoilwtico,...
```

**Auxiliary files** (if using merged dataset):
- `data/raw/stores.csv` - Store metadata
- `data/raw/holidays_events.csv` - Holiday calendar
- `data/raw/oil.csv` - Oil prices

If using sample data:
```bash
python create_drift_test_data.py  # Generates drift_test.csv for testing
```

### 5. Initialize Django Database

```bash
cd django_app
python manage.py migrate
```

---

## Running the Application

### Single Terminal (Development Mode)

**Start Both Servers**:
```bash
python dev.py run-all
```

This will:
1. Start FastAPI backend on `http://localhost:8000`
2. Start Django frontend on `http://localhost:8080`
3. Display URLs for dashboard, API docs, and endpoints

**Output**:
```
======================================================================
🚀 OmniFlow - Starting Frontend & Backend Servers
======================================================================

[1/2] Starting FastAPI Backend (port 8000)...
[2/2] Starting Django Frontend (port 8080)...

======================================================================
✅ Both servers started successfully!
======================================================================

🌐 Access the application:
   • Web Dashboard:    http://localhost:8080
   • API Docs:         http://localhost:8000/docs
   • API ReDoc:        http://localhost:8000/redoc

📊 Available Pages:
   • Dashboard:        http://localhost:8080/
   • Train Model:      http://localhost:8080/upload/
   • Make Prediction:  http://localhost:8080/predict/
   • Drift & Retrain:  http://localhost:8080/drift/

======================================================================
Press Ctrl+C in each terminal window to stop servers
======================================================================
```

### Multiple Terminals (Advanced)

**Terminal 1 - FastAPI Backend**:
```bash
python dev.py run-api
```

**Terminal 2 - Django Frontend**:
```bash
python dev.py run-web
```

**Terminal 3 - Train Initial Model**:
```bash
python dev.py run-train
```

---

## Accessing the Application

Once servers are running:

### Web Dashboard
Open http://localhost:8080

**Pages**:
- **Home** (`/`) - Overview and quick links
- **Train** (`/upload/`) - Upload CSV and train model
- **Predict** (`/predict/`) - Single item prediction
- **Drift Analysis** (`/drift/`) - Upload new data, detect drift, auto-retrain

### API Documentation

**FastAPI Interactive Docs**:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Try It Out**:
```bash
curl -X GET http://localhost:8000/health
```

---

## File Structure Quick Reference

```
OmniFlow-Sales-AI/
├── api/                          # FastAPI backend
│   ├── main.py                   # Core API endpoints & job orchestration
│   └── __init__.py
├── django_app/                   # Django frontend
│   ├── manage.py                 # Django management
│   ├── omni_flow/                # Django project settings
│   │   ├── settings.py           # Django configuration
│   │   ├── urls.py               # Project URL routing
│   │   └── wsgi.py               # WSGI application
│   └── omniapp/                  # Django app
│       ├── views.py              # HTTP request handlers
│       ├── urls.py               # App-level routing
│       └── templates/            # HTML templates
│           ├── index.html        # Dashboard
│           ├── upload.html       # Training interface
│           ├── train_result.html # Results page
│           ├── drift_result.html # Drift analysis
│           └── predict.html      # Prediction interface
│
├── src/                          # ML pipeline
│   ├── training.py               # Model training & validation
│   ├── preprocessing.py          # Feature engineering
│   ├── drift.py                  # Drift detection
│   ├── retrain.py                # Retraining logic
│   ├── ingestion.py              # Data loading
│   ├── inference.py              # Predictions
│   └── utils.py                  # Helper utilities
│
├── data/                         # Data directory (git-ignored)
│   ├── raw/                      # Raw datasets
│   │   ├── combined.csv          # Training data
│   │   ├── stores.csv            # Store metadata
│   │   ├── holidays_events.csv   # Holidays
│   │   └── oil.csv               # Oil prices
│   └── processed/                # Processed data
│
├── models/                       # Model artifacts (git-ignored)
│   ├── champion.joblib           # Trained model
│   ├── champion_metrics.json     # Baseline metrics
│   └── reference_stats.json      # Drift baseline
│
├── docs/                         # Documentation
│   ├── 01-ARCHITECTURE.md        # System design
│   ├── 02-GETTING_STARTED.md     # This file
│   ├── 03-API_REFERENCE.md       # Endpoint docs
│   ├── 04-ML_PIPELINE.md         # ML details
│   ├── 05-DEPLOYMENT.md          # Production setup
│   └── 06-CONTRIBUTING.md        # Contribution guide
│
├── dev.py                        # Development CLI
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── README.md                     # Main documentation
└── CHANGELOG.md                  # Version history
```

---

## Troubleshooting

### Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Kill process on port 8000
lsof -ti :8000 | xargs kill -9    # macOS/Linux
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force  # Windows
```

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Ensure you're in project root
cd OmniFlow-Sales-AI
python dev.py run-all
```

### Data File Not Found

**Error**: `FileNotFoundError: data/raw/combined.csv`

**Solution**:
```bash
# Generate sample data
python create_drift_test_data.py
```

### Django Migration Issues

```bash
cd django_app
python manage.py migrate --run-syncdb
python manage.py runserver 0.0.0.0:8080
```

---

## Next Steps

1. **Use Training Interface**: http://localhost:8080/upload/
2. **Check Training Progress**: Real-time live bar with percentage
3. **View Metrics**: After training, see R², MAE, RMSE, and model info
4. **Try Predictions**: http://localhost:8080/predict/
5. **Test Drift**: http://localhost:8080/drift/ with new data
6. **Monitor Auto-Retrain**: System automatically decides to retrain or keep champion

---

## Performance Tips

- **First Training**: Takes 2-5 minutes with large datasets
- **Subsequent Drifts**: Faster (~30s) as reference stats are cached
- **Predictions**: <100ms per request
- **Memory**: Monitor RAM if running on 8GB machines

---

## Support

For issues or questions:
1. Check [API Reference](03-API_REFERENCE.md)
2. Review [ML Pipeline](04-ML_PIPELINE.md)
3. See [Architecture](01-ARCHITECTURE.md) for design details

