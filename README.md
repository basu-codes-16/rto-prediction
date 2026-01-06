# RTO Prediction System

Production-grade machine learning pipeline for predicting Return-to-Origin (RTO) probability in e-commerce logistics.

## 🎯 Project Overview

**Objective:** Build an ML system that predicts whether an e-commerce order will be returned to the seller (RTO) based on historical data, geographic patterns, and transactional features.


---

## 📊 Dataset

| Dataset | Records | Usage |
|---------|---------|-------|
| Meesho Orders | 208 | Primary training data |
| Meesho Forward Reports | 138 | RTO labels (order_status) |
| DTDC Courier | 49,639 | (Future use for enrichment) |
| Delhivery | 144,867 | (Future use for enrichment) |

**Final Training Set:** 130 labeled orders (74 delivered, 56 RTO)

### Key Assumptions:
1. **RTO Definition:** Orders with status 'return', 'rto', or 'cancelled' = RTO (label: 1)
2. **Dataset Limitation:** Only Meesho data used due to lack of join keys with courier datasets
3. **Class Balance:** 57% delivered vs 43% RTO (mild imbalance, handled with SMOTE)

---

## 🏗️ Architecture

```
rto-prediction/
├── data/
│   ├── raw/                    # Original CSVs
│   ├── processed/              # Cleaned & featured data
│   ├── external/               # Pincode database (optional)
│   └── rto_data.db            # SQLite database
├── src/
│   ├── data/
│   │   ├── sql_ingestion.py   # CSV → SQL pipeline
│   │   └── label_creator.py   # Create RTO labels
│   ├── features/
│   │   ├── build_features.py  # Feature engineering
│   │   └── address_quality_nlp.py  # NLP address scoring (for future use)
│   ├── models/
│   │   ├── train_model.py     # Model training (MLflow)
│   │   └── validate_model.py  # Cross-validation
│   ├── api/
│   │   └── main.py            # FastAPI inference service
│   ├── dashboard/
│   │   └── app.py             # Streamlit dashboard
│   └── utils.py               # Shared utilities
├── tests/
│   ├── test_features.py       # Unit tests
│   └── test_api.py            # API integration tests
├── models/production/         # Saved model artifacts
├── mlruns/                    # MLflow experiment tracking
├── configs/
│   └── config.yaml            # Configuration
└── notebooks/                 # EDA notebooks

```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv rto_venv
source rto_venv/bin/activate  # Windows: rto_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Ingestion

```bash
# Ingest CSVs into SQLite
python src/data/sql_ingestion.py
```

### 3. Create Labels & Features

```bash
# Create RTO labels from order_status
python src/data/label_creator.py

# Build features (temporal, geographic, transactional, NLP)
python src/features/build_features.py
```

### 4. Train Models

```bash
# Train 4 models with MLflow tracking
python src/models/train_model.py

# Validate with cross-validation
python src/models/validate_model.py
```

### 5. Run Dashboard

```bash
streamlit run src/dashboard/app.py
```

### 6. Deploy API

```bash
# Start API server
python src/api/main.py

# Test API
python tests/test_api.py

# Interactive docs at http://localhost:8000/docs
```

---

## 📈 Model Performance

| Model | Accuracy | F1-Score | ROC-AUC | CV F1 (5-Fold) |
|-------|----------|----------|---------|----------------|
| Logistic Regression | **92.3%** | **90.9%** | **99.4%** | 94.6% ± 3.5% |
| Random Forest | 92.3% | 90.9% | 99.4% | 94.6% ± 3.5% |
| Gradient Boosting | 92.3% | 90.9% | 99.4% | 94.6% ± 3.5% |
| XGBoost | 92.3% | 90.9% | 99.4% | 94.6% ± 3.5% |
| LightGBM | 92.3% | 90.9% | 99.4% | 94.6% ± 3.5% |

**Best Model:** Logistic Regression (simplest, equally performant)

### Class Imbalance Handling:
- **SMOTE:** Balanced training set from 45:59 to 59:59
- **Class Weights:** Applied in Random Forest
- **Evaluation:** Used F1-score, PR-AUC (not just accuracy)

---

## 🔧 Feature Engineering

### Feature Categories (27 features):

1. **Temporal (6):** day_of_week, is_weekend, month, is_month_start, is_month_end
2. **Geographic (5):** is_metro, pin_rto_rate, state_rto_rate, pin_order_count, state_order_count
3. **Transactional (8):** price_per_unit, discount_pct, shipping_to_price_ratio, quantity, prices
4. **Address Quality (3):** has_valid_pincode, has_state, address_quality_score
5. **Product (5):** product_rto_rate, product_order_count, product_length

### Top 5 Predictive Features:
1. `pin_rto_rate` - Historical RTO rate by pincode
2. `state_rto_rate` - Historical RTO rate by state
3. `final_price` - Order value
4. `is_metro` - Metro vs non-metro city
5. `product_rto_rate` - Product category RTO history

---

## 🧪 NLP for Address Quality

**Approach:** Rule-based pattern matching (regex)

**Components Detected:**
- Building number: `\b\d+[/-]?\d*\b`
- Pincode: `\b\d{6}\b`
- Landmark: `near|opposite|beside`
- Road/Street: `road|rd|street|lane`
- Area: `sector|colony|nagar`

**Score Calculation:**
```python
completeness = 0.25 * has_building + 0.25 * has_pincode + 
               0.15 * has_road + 0.15 * has_area + 
               0.10 * has_landmark + 0.10 * has_city
```

---

## 🔄 MLOps Pipeline

### Experiment Tracking (MLflow):
- 5 model runs logged
- Hyperparameters versioned
- Metrics tracked (accuracy, F1, ROC-AUC, PR-AUC)
- Model artifacts saved

```bash
# View MLflow UI
mlflow ui --port 5000
```

### Model Deployment:
- Saved to `models/production/model.pkl`
- Scaler and encoders versioned
- Metadata JSON with performance metrics
- FastAPI for inference

### Reproducibility:
- Random seed: 42
- Config-driven training
- Environment: `requirements.txt`

---

## 📡 API Usage

### Predict Single Order:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "day_of_week": 2,
    "is_metro": 1,
    "pin_rto_rate": 0.25,
    "state_rto_rate": 0.35,
    "final_price": 1200,
    ...
  }'
```

### Response:

```json
{
  "rto_probability": 0.15,
  "rto_prediction": 0,
  "risk_level": "Low",
  "confidence": 0.70
}
```

---

## 🎨 Dashboard Features

- **Overview:** RTO distribution, temporal patterns
- **Geographic:** State-wise RTO heatmap, metro vs non-metro
- **Model Performance:** Confusion matrix, feature importance
- **Predictions:** Sample order analysis

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Test API
python tests/test_api.py

# Test high-risk scenarios
python tests/test_high_risk.py
```

---

## ⚠️ Limitations & Future Work

### Current Limitations:
1. **Small Dataset:** Only 130 training samples (low generalization)
2. **DTDC/Delhivery Unused:** No join keys, no RTO labels
3. **Basic NLP:** Rule-based, not semantic understanding
4. **No COD Flag:** Payment mode data missing

### Proposed Improvements:
1. **Data Enrichment:** 
   - Use DTDC/Delhivery for pincode-level features
   - External pincode database (lat/long, urban/rural)
2. **Advanced NLP:**
   - Transformer models (BERT) for address parsing
   - Geocoding API for address validation
3. **Model Enhancements:**
   - Ensemble methods (stacking)
   - Time-series features (order history)
4. **Production Monitoring:**
   - Data drift detection
   - Model retraining pipeline
   - A/B testing framework

---

## 📚 References

- [SMOTE for Imbalanced Learning](https://arxiv.org/abs/1106.1813)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)

---

## 👤 Author

**Assignment for IIT Delhi Research Role**  
Date: January 2026