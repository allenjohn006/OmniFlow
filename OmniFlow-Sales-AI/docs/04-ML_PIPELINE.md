# 🧠 ML Pipeline Documentation

## Overview

The ML pipeline is a production-grade preprocessing and training system built on scikit-learn with XGBoost. It handles 3M+ row datasets efficiently with explicit dtype management for pandas 2.2+ compatibility.

---

## Data Lifecycle

### 1. Data Ingestion (`src/ingestion.py`)

**Purpose**: Load and validate raw data from multiple sources

**Process**:
```python
from src.ingestion import load_data

# Load from CSV
df = load_data("data/raw/train.csv")

# Load from bytes (uploaded file)
df = load_data(file_bytes)
```

**Input**:
- CSV files with sales transactions
- Auxiliary data (stores, holidays, oil prices)

**Output**:
- Pandas DataFrame with raw data
- Validation checks for required columns

**Key Operations**:
- Date parsing with error coercion
- Missing value detection
- Column name normalization
- Data type inference

---

### 2. Preprocessing & Feature Engineering (`src/preprocessing.py`)

**Purpose**: Transform raw data into ML-ready feature matrix

**Core Function**: `build_training_frame()`

**Input**: Raw DataFrame with columns like:
```
date, store_nbr, family, sales, onpromotion, cluster, dcoilwtico, ...
```

**Processing Steps**:

#### Step 1: Date Extraction
```python
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek
```

#### Step 2: Binary Indicators
```python
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['is_salary_day'] = (df['date'].dt.day == 15).astype(int)
```

#### Step 3: Holiday Categorization
- Load holiday calendar
- Classify each date as holiday type or "None"

#### Step 4: Missing Value Handling
```python
df['dcoilwtico'].fillna(df['dcoilwtico'].mean(), inplace=True)
df['categorical_col'].fillna('None', inplace=True)
```

#### Step 5: Feature Type Validation
```python
# Ensure all numerical features are float64
NUMERICAL_FEATURES = [...]
df[NUMERICAL_FEATURES] = df[NUMERICAL_FEATURES].astype('float64')
```

**Output**:
- Feature matrix X with 15 features
- Target vector y
- Training/test split (pre-2016 | post-2016)

**Feature List**:

**Categorical** (5 features):
- `family` - Product family
- `city` - Store city
- `state` - Store state
- `type` - Store type
- `holiday_type` - Holiday classification

**Numerical** (10 features):
- `store_nbr` - Store ID
- `cluster` - Store cluster
- `onpromotion` - Promotion indicator
- `dcoilwtico` - Oil price
- `lag_7` - 7-day sales lag
- `year`, `month`, `dayofweek` - Date components
- `is_weekend`, `is_salary_day` - Binary indicators

**Data Split**:
```
Training: date < 2016-01-01 (2.5M samples)
Test:     date >= 2016-01-01 (500K samples)
```

---

### 3. Model Pipeline (`src/training.py`)

**Purpose**: Build and train the ML model

**Architecture**: scikit-learn Pipeline

```python
Pipeline(
    steps=[
        ("preprocessor", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numerical", "passthrough", NUMERICAL_FEATURES)
        ])),
        ("model", XGBRegressor(...))
    ]
)
```

**Preprocessing Stage**:
- **Categorical**: One-Hot Encoding
  - Creates binary columns for each category
  - Handles unknown categories gracefully
- **Numerical**: Pass-through (no transformation)
  - Already normalized by preprocessing

**Model Stage**: XGBoost Regressor

**Hyperparameters**:
```python
XGBRegressor(
    n_estimators=300,           # 300 boosting rounds
    max_depth=8,                # Tree depth
    learning_rate=0.05,         # Shrinkage parameter
    subsample=0.9,              # Row sampling ratio
    colsample_bytree=0.9,       # Column sampling ratio
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1                   # Use all CPU cores
)
```

**Progress Callbacks**:
While training, 8-phase progress tracking update via HTTP:

1. **Preprocessing** (0-10%) - Loading and feature engineering
2. **Encoding** (10-15%) - One-hot encoding features
3. **Training** (15-95%) - Boosting iterations
4. **Validation** (95-98%) - Test set evaluation
5. **Metrics** (98-99%) - Calculate R², MAE, RMSE
6. **Saving** (99-100%) - Persist model and statistics
7. **Done** (100%) - Complete

---

### 4. Metrics Calculation

**Metrics Computed**:

#### R² Score (Coefficient of Determination)
$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$

- Range: [0, 1] (higher is better)
- Interpretation: Percent of variance explained by model
- Example: R² = 0.88 means model explains 88% of sales variance

#### Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- Unit: Same as target (sales amount)
- Interpretation: Average prediction error
- Example: MAE = 103.52 means avg error is $103.52

#### Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

- Unit: Same as target
- Interpretation: Penalizes large errors more than MAE
- Example: RMSE = 145.23 means typical error accounting for outliers

**Baseline Metrics** (saved in `models/champion_metrics.json`):
```json
{
  "r2_score": 0.8805,
  "mae": 103.52,
  "rmse": 145.23,
  "train_samples": 2500000,
  "test_samples": 500000,
  "features_count": 15
}
```

---

## Drift Detection (`src/drift.py`)

### Purpose
Monitor for data distribution shifts (drift) that might indicate model degradation is needed.

### Drift Strategy: Statistical Hypothesis Testing

**Key Insight**: Drift ≠ Performance Degradation
- Drift: Distribution of input features changed
- Performance: Model predictions became inaccurate
- These are NOT always correlated!

### Process

#### 1. Load Reference Statistics
From `models/reference_stats.json` (saved at training time):
```json
{
  "numerical_stats": {
    "store_nbr": {"mean": 51.5, "std": 29.2},
    "dcoilwtico": {"mean": 92.1, "std": 15.3},
    ...
  },
  "categorical_stats": {
    "family": {"count": {"AUTOMOTIVE": 0.15, "PRODUCE": 0.25, ...}},
    ...
  }
}
```

#### 2. Run Statistical Tests

**For Numerical Features**: Kolmogorov-Smirnov (KS) Test
```
H0: New data comes from same distribution as reference
HA: New data comes from different distribution

KS Statistic: Maximum difference between CDFs
p-value > 0.05: Accept H0 (no drift)
p-value ≤ 0.05: Reject H0 (drift detected)
```

**For Categorical Features**: Chi-Square Test
```
H0: Category proportions unchanged
HA: Category proportions changed

Chi² Statistic: Measure of proportion distance
p-value > 0.05: Accept H0 (no drift)
p-value ≤ 0.05: Reject H0 (drift detected)
```

#### 3. Calculate Results

**Per-Feature**:
```json
{
  "feature": "dcoilwtico",
  "feature_type": "numerical",
  "p_value": 0.0023,
  "statistic": 0.152,
  "is_drifted": true,
  "reason": "KS test p-value 0.0023 < 0.05"
}
```

**Overall**:
```json
{
  "drift_detected": true,
  "drift_ratio": 0.7333,
  "drifted_features": 11,
  "total_features": 15,
  "drifted_feature_names": ["dcoilwtico", "onpromotion", "family", ...]
}
```

---

## Auto-Retrain Decision Logic (`src/retrain.py`)

### Problem
**When should the model be retrained?**
- Always? (model churn, compute waste)
- Never? (performance degradation)
- Smart? (only when needed)

### Solution: Guardrail-Based Decision Engine

#### Decision Tree

```
┌─ Drift Detected? ─────────────────────┐
│                                       │
│  NO                                YES│
│  ↓                                    ↓
│ [Keep]                       [Evaluate on new data]
│                                    ↓
└────────────────┬──────────────────────┘
                 ↓
        R² drops > 5% ?
               ↙      ↘
            YES        NO
             ↓          ↓
         [Check MAE/RMSE if YES → RETRAIN]
             ↓
        MAE or RMSE
        increase > 10% ?
             ↙     ↘
          YES       NO
           ↓         ↓
       [RETRAIN]  [CHECK drift_ratio]
                       ↓
                  > 75% drift?
                   ↙      ↘
                YES        NO
                 ↓          ↓
             [RETRAIN]   [KEEP]
```

#### Guardrails (Thresholds)

| Metric | Threshold | Action |
|--------|-----------|--------|
| R² drop | > 5% | Retrain |
| MAE increase | > 10% | Retrain |
| RMSE increase | > 10% | Retrain |
| Drift ratio | > 75% | Retrain |

**Examples**:
- R² baseline 0.88 → new 0.85 (3.4% drop) → Keep
- R² baseline 0.88 → new 0.82 (6.8% drop) → Retrain
- 75% features drifted → Retrain

#### Challenger Evaluation

If retrain is triggered:

1. **Train Challenger**:
   - New model trained on new data
   - Same pipeline, same hyperparameters

2. **Evaluate Both**:
   - Champion on new data: R²_c, MAE_c, RMSE_c
   - Challenger on new data: R²_new, MAE_new, RMSE_new

3. **Compare**:
   ```python
   if R²_new > R²_c and MAE_new < MAE_c:
       promote_challenger()  # Update champion.joblib
   else:
       keep_champion()       # No change
   ```

---

## Reference Statistics (`models/reference_stats.json`)

Saved during initial training, used for drift detection:

```json
{
  "numerical_stats": {
    "store_nbr": {
      "mean": 51.5,
      "std": 29.2
    },
    "onpromotion": {
      "mean": 0.08,
      "std": 0.27
    },
    "cluster": {
      "mean": 5.2,
      "std": 3.1
    },
    "dcoilwtico": {
      "mean": 92.1,
      "std": 15.3
    },
    "lag_7": {
      "mean": 2150,
      "std": 850
    },
    "year": {
      "mean": 2015.3,
      "std": 0.46
    }
  },
  "categorical_stats": {
    "family": {
      "count": {
        "AUTOMOTIVE": 0.15,
        "BABY CARE": 0.08,
        "BEAUTY": 0.06,
        ...
      },
      "unique": 33
    },
    "city": {
      "count": { ... },
      "unique": 22
    }
  },
  "training_info": {
    "samples": 2500000,
    "features": 15,
    "trained_at": "2026-04-01T10:00:00Z"
  }
}
```

---

## Prediction Inference (`src/inference.py`)

**Purpose**: Generate predictions for new unseen data

**Process**:

1. **Load Champion Model**:
   ```python
   model = joblib.load("models/champion.joblib")
   ```

2. **Validate Input Features**:
   ```python
   - Check all required features present
   - Check feature types match training
   - Ensure categorical values are known (handled by encoder)
   ```

3. **Apply Same Preprocessing**:
   - Pipeline automatically applies preprocessing
   - One-hot encoding (unknown → all zeros)
   - Numerical pass-through

4. **Generate Prediction**:
   ```python
   prediction = model.predict(X)[0]
   confidence = model.predict_proba(X) if available
   ```

5. **Return Result**:
   ```json
   {
     "prediction": 1852.45,
     "confidence_interval": {
       "lower": 1750.23,
       "upper": 1954.67
     }
   }
   ```

---

## Performance Considerations

### Training Time
- **Small dataset** (500K rows): ~30 seconds
- **Medium dataset** (1-2M rows): ~60-120 seconds
- **Large dataset** (3M+ rows): ~3-5 minutes

**Optimization Strategies**:
- Increase `n_jobs=-1` (uses all CPU cores)
- Reduce `n_estimators` (fewer boosting rounds)
- Use GPU: XGBoost supports `tree_method='gpu_hist'`

### Memory Usage
- **Baseline**: ~2GB for 3M rows dataset
- **Peak during training**: ~4-6GB
- **After training**: ~500MB (trained model)

**Optimization**:
- Stream data in chunks for very large datasets
- Use categorical dtype (not string) to save memory

### Inference Latency
- **Single prediction**: <50ms
- **Batch (1000 rows)**: ~200ms

---

## Quality Assurance

### Model Validation Checks

1. **Schema Validation**:
   ```python
   Expected columns = [store_nbr, family, city, ...]
   Actual columns must match exactly
   ```

2. **Type Validation**:
   ```python
   All numerical features: float64
   All categorical features: str or object
   ```

3. **Range Validation**:
   ```python
   store_nbr: 1-54
   cluster: 1-17
   dcoilwtico: positive
   ```

4. **Sanity Checks**:
   ```python
   Predictions: 0-10000 (sales range)
   R² score: 0-1
   MAE: >0
   ```

---

## Future Improvements

- [ ] Feature importance tracking
- [ ] Hyperparameter auto-tuning (Bayesian optimization)
- [ ] Ensemble models (blend XGBoost + LightGBM)
- [ ] SHAP values for model explanation
- [ ] Automated feature selection
- [ ] Multi-step ahead forecasting
- [ ] Seasonality detection and adjustment

