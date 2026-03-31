# OmniFlow Sales AI

A comprehensive AI/ML solution for sales forecasting and monitoring with data drift detection.

## Project Structure

- **data/**: Raw and processed data
  - `raw/`: Original datasets
  - `processed/`: Cleaned and preprocessed data ready for training
  
- **models/**: Trained models and reference statistics
  - `model_champion.pkl`: Production model
  - `reference_stats.json`: Reference statistics for drift detection
  - `columns.pkl`: Feature column mappings

- **src/**: Core ML modules
  - `ingestion.py`: Data loading
  - `preprocessing.py`: Data cleaning and feature engineering
  - `training.py`: Model training
  - `drift.py`: Data drift detection
  - `retrain.py`: Model retraining logic
  - `utils.py`: Utility functions

- **api/**: FastAPI application
  - `main.py`: Main API entry point
  - `routes/`: API endpoints for training, predictions, and drift detection

- **django_app/**: Django web application
  - `manage.py`: Django management script
  - `omniapp/`: Main app with views, urls, templates
  - `templates/`: HTML templates for upload, results

- **mlruns/**: MLflow experiment tracking

- **powerbi/**: Power BI dashboards

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python src/training.py
```

### Running API
```bash
uvicorn api.main:app --reload
```

### Running Django App
```bash
python django_app/manage.py runserver
```

## Data Drift Detection

Monitor model performance with drift detection:
```bash
python src/drift.py
```

## License

MIT
