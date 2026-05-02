# ⬡ CreditIQ — Loan Default Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-AUC%200.786-success?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=flat-square)

**An end-to-end credit risk scoring system trained on 307,511 real loan applications.**  
Predicts default probability in real time · Explains every decision with SHAP · Deployed via Docker on Render.

[Live Demo](#deployment) · [API Docs](#api-reference) · [Model Results](#model-performance)

</div>

---

## What This System Does

A loan applicant submits a request. Within milliseconds this system:

1. Takes their financial profile as input
2. Runs it through a LightGBM model trained on 307,511 real-world applications
3. Returns a default probability score between 0 and 1
4. Explains exactly which factors drove the decision using SHAP values
5. Issues a verdict — **APPROVE / REVIEW / REJECT** — with thresholds calibrated to banking standards

This is not a tutorial project. It replicates what production credit risk teams at banks actually build — from raw messy data through feature engineering, model training, explainability, API serving, and live deployment.

---

## Model Performance

| Metric | Baseline (Logistic Regression) | Final (LightGBM) |
|---|---|---|
| AUC-ROC | 0.7653 | **0.7864** |
| KS Statistic | — | **0.4888** |
| CV Strategy | 3-Fold Stratified | 5-Fold Stratified |
| Class Handling | `class_weight='balanced'` | `class_weight='balanced'` |

**What these numbers mean:**

- **AUC-ROC 0.786** — The model correctly ranks a random defaulter above a random non-defaulter 78.6% of the time. A random classifier scores 0.5. Industry-grade retail credit models typically range from 0.72 to 0.82.
- **KS Statistic 0.489** — Measures the maximum separation between the default and non-default score distributions. Anything above 0.40 is considered a strong model by credit risk teams. 0.489 is production-ready.

---

## Why These Choices

### Why LightGBM over XGBoost or Random Forest?

- **Missing value handling** — LightGBM handles NaN natively by learning the optimal split direction for missing values. This dataset has 67 columns with missing values. No imputation needed.
- **Speed** — LightGBM uses histogram-based leaf-wise tree growth. On 307k rows with 210 features it trains ~4x faster than XGBoost.
- **Memory efficiency** — Critical on an 8GB RAM machine with supplementary tables totalling 2GB+.
- **Kaggle validation** — Top solutions on this exact Home Credit dataset use LightGBM. The architecture is battle-tested.

### Why Optuna over GridSearchCV?

- **Bayesian optimization** — Optuna's TPE sampler learns from previous trials and focuses search on promising hyperparameter regions. GridSearch wastes compute on bad regions.
- **Early stopping integration** — Each trial uses LightGBM early stopping. Bad configurations exit fast, good ones run fully.
- **Modern standard** — Optuna is what production ML teams use. GridSearch is a tutorial artifact.

### Why SHAP for explainability?

- **SR 11-7 compliance** — The Federal Reserve's model risk management guidelines require banks to explain model decisions to regulators. SHAP provides mathematically rigorous feature attributions.
- **TreeExplainer** — Uses exact Shapley values for tree-based models. Fast and exact, not approximate.
- **Business interpretability** — A loan officer can read "EXT_SOURCE_2 decreased your risk by 0.45" and understand it. Black-box predictions are legally unusable in credit decisions.

### Why KS Statistic alongside AUC?

AUC is the ML standard but banks use KS. The KS statistic measures how well the model separates defaulters from non-defaulters at the optimal threshold. Credit risk teams in India (CIBIL scoring) and globally use KS as the primary scorecard validation metric. Reporting both shows you understand the domain, not just the algorithm.

---

## Dataset

**Source:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)

Real loan application data from Home Credit, an international consumer finance provider. Used with permission under Kaggle competition terms.

| File | Rows | Columns | Description |
|---|---|---|---|
| `application_train.csv` | 307,511 | 122 | Main table — one row per loan application |
| `application_test.csv` | 48,744 | 121 | Test set without target |
| `bureau.csv` | 1,716,428 | 17 | Credit history from other institutions |
| `bureau_balance.csv` | 27,299,925 | 3 | Monthly balances of bureau credits |
| `previous_application.csv` | 1,670,214 | 37 | Previous loan applications at Home Credit |
| `installments_payments.csv` | 13,605,401 | 8 | Repayment history |
| `credit_card_balance.csv` | 3,840,312 | 23 | Credit card monthly balances |
| `POS_CASH_balance.csv` | 10,001,358 | 8 | POS and cash loan monthly balances |

**Target variable:** `TARGET` — binary, 1 = defaulted, 0 = repaid  
**Class imbalance:** 91.9% non-default vs 8.1% default (11.4:1 ratio)

---

## Project Structure

```
loan-default-prediction/
│
├── data/
│   ├── raw/                        # Kaggle CSVs (gitignored — too large)
│   └── processed/                  # Parquet feature store
│
├── notebooks/
│   └── 01_eda.ipynb                # Full exploratory data analysis
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Memory-optimized CSV loading
│   ├── feature_engineering.py      # All feature engineering + aggregations
│   ├── train.py                    # Training pipeline with MLflow + Optuna
│   └── predict.py                  # Inference + SHAP explanation
│
├── api/
│   ├── __init__.py
│   └── main.py                     # FastAPI REST API
│
├── app/
│   └── streamlit_app.py            # Dark-themed banking dashboard
│
├── models/
│   └── lgbm_model.pkl              # Trained model artifact (gitignored)
│
├── reports/
│   ├── target_distribution.png
│   ├── missing_values.png
│   ├── ext_source_analysis.png
│   ├── categorical_analysis.png
│   └── days_employed_anomaly.png
│
├── Dockerfile                      # Single container — FastAPI + Streamlit
├── start.sh                        # Container startup script
├── render.yaml                     # Render deployment config
├── requirements.txt                # All Python dependencies pinned
└── README.md
```

---

## Feature Engineering

We go from 122 raw features to **210 engineered features** across 7 tables.

### Main Application Table

| Feature | Formula | Why It Matters |
|---|---|---|
| `CREDIT_INCOME_RATIO` | `AMT_CREDIT / AMT_INCOME_TOTAL` | Core affordability metric used in all credit scoring |
| `ANNUITY_INCOME_RATIO` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | Monthly burden as fraction of income |
| `CREDIT_TERM` | `AMT_ANNUITY / AMT_CREDIT` | Loan duration proxy |
| `GOODS_CREDIT_RATIO` | `AMT_GOODS_PRICE / AMT_CREDIT` | Loan-to-value ratio |
| `EXT_SOURCE_MEAN` | Mean of EXT_SOURCE_1,2,3 | Combined external credit score |
| `EXT_SOURCE_STD` | Std of EXT_SOURCE_1,2,3 | Score consistency across bureaus |
| `AGE_YEARS` | `-DAYS_BIRTH / 365` | Human-readable age |
| `EMPLOYED_YEARS` | `-DAYS_EMPLOYED / 365` | Employment tenure |
| `DAYS_EMPLOYED_ANOMALY` | Flag where DAYS_EMPLOYED == 365243 | 18% of rows have fake employment value |
| `{col}_MISSING` | Binary flag for 6 high-missing columns | Missingness itself is predictive |

### Bureau Aggregations (per customer)

Aggregated from 1.7M credit bureau records and 27M monthly balance records:
- Average and total outstanding debt
- Maximum days past due across all credits
- Count of active vs closed credits
- Number of credit prolongations

### Previous Application Aggregations

Aggregated from 1.6M previous loan applications:
- Approval and refusal counts — **number of previous rejections is highly predictive**
- Credit utilization in previous loans
- Average loan amounts and terms

### Installment Payment Aggregations

Aggregated from 13.6M payment records:
- `PAYMENT_DIFF` — how much more or less than due was paid
- `DAYS_LATE` — days past due per installment
- Total late payment history

### Data Quality Issue — DAYS_EMPLOYED Anomaly

18% of applicants (55,374 rows) have `DAYS_EMPLOYED = 365,243` — a sentinel value meaning "not employed" or "unknown." A naive model treats this as a real number (1,000 years employed) and breaks completely.

**Our fix:**
1. Replace 365,243 with `NaN`
2. Create a binary flag column `DAYS_EMPLOYED_ANOMALY = 1`

The model then learns that the anomaly flag itself is predictive — these rows have a 5.4% default rate vs 8.7% for normal rows, which means they're lower risk (mostly pensioners and students).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                           │
│  Kaggle CSVs → reduce_memory() → Feature Store         │
│  7 tables · 57M+ total rows · 210 engineered features  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   ML PIPELINE                           │
│  EDA → Feature Engineering → SMOTE → Model Training    │
│  LogReg baseline → Optuna (30 trials) → LightGBM       │
│  MLflow experiment tracking → SHAP explainability       │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  SERVING LAYER                          │
│  FastAPI (port 8000) ← POST /predict                   │
│  Streamlit Dashboard (port 8501)                        │
│  Docker container → Render deployment                   │
└─────────────────────────────────────────────────────────┘
```

---

## API Reference

**Base URL (local):** `http://localhost:8000`  
**Interactive docs:** `http://localhost:8000/docs`

### `GET /health`

Returns model status and feature count.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "n_features": 208
}
```

### `POST /predict`

Predict default probability for a single applicant.

**Request body (all fields optional — model handles missing values natively):**

```json
{
  "EXT_SOURCE_1": 0.65,
  "EXT_SOURCE_2": 0.70,
  "EXT_SOURCE_3": 0.68,
  "AMT_INCOME_TOTAL": 180000,
  "AMT_CREDIT": 450000,
  "AMT_ANNUITY": 22000,
  "AGE_YEARS": 38.4,
  "EMPLOYED_YEARS": 6.8,
  "CREDIT_INCOME_RATIO": 2.5
}
```

**Response:**

```json
{
  "risk_score": 0.0399,
  "risk_percent": 3.99,
  "verdict": {
    "label": "APPROVE",
    "color": "green",
    "description": "Low risk — recommend approval"
  },
  "explanations": [
    {
      "feature": "EXT_SOURCE_MEAN",
      "value": 0.676,
      "shap_value": -0.4988,
      "direction": "decreases_risk"
    }
  ]
}
```

**Verdict thresholds:**

| Score | Verdict | Meaning |
|---|---|---|
| 0.00 – 0.30 | ✅ APPROVE | Low risk |
| 0.30 – 0.50 | ⚠️ REVIEW | Manual review recommended |
| 0.50 – 1.00 | ❌ REJECT | High risk |

### `POST /predict/batch`

Predict for up to 100 applicants in a single request. Accepts an array of the same request body as `/predict`.

---

## Running Locally

### Prerequisites

- Python 3.11+
- 8GB RAM minimum
- ~3GB free disk space for dataset

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/loan-default-prediction.git
cd loan-default-prediction

# Create and activate virtual environment
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
source venv/bin/activate          # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

1. Go to [kaggle.com/competitions/home-credit-default-risk](https://www.kaggle.com/competitions/home-credit-default-risk)
2. Download all files → extract into `data/raw/`

### Train the Model

```bash
python -m src.train
```

This runs the full pipeline:
- Loads and aggregates all 7 tables
- Engineers 210 features
- Trains logistic regression baseline
- Runs 30 Optuna trials for LightGBM
- Saves model to `models/lgbm_model.pkl`
- Logs all metrics to MLflow

Takes approximately 2-3 hours on a standard laptop.

### View MLflow Experiments

```bash
mlflow ui
# Open http://127.0.0.1:5000
```

### Start the API

```bash
python -m uvicorn api.main:app --reload --port 8000
# Docs at http://127.0.0.1:8000/docs
```

### Start the Dashboard

```bash
streamlit run app/streamlit_app.py
# Opens at http://localhost:8501
```

---

## Deployment

### Docker (local)

```bash
docker build -t creditiq .
docker run -p 8000:8000 -p 8501:8501 creditiq
```

### Render (live)

1. Push to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and deploys

The `render.yaml` configures:
- Docker runtime
- Auto-deploy on every push to `main`
- Health check on `/health`

---

## Key Concepts Explained

**Stratified K-Fold Cross Validation**  
With 11.4:1 class imbalance, random splits can put almost no defaulters in a validation fold. Stratified splitting preserves the 8.1% default rate in every fold, giving honest performance estimates.

**SMOTE (Synthetic Minority Oversampling Technique)**  
Creates synthetic defaulter samples by interpolating between existing defaulter feature vectors. Used to balance training data without simply duplicating rows.

**SR 11-7 Model Risk Management**  
Federal Reserve guidance requiring banks to validate, document, and explain all models used in credit decisions. Explainability via SHAP directly addresses the "conceptual soundness" requirement of SR 11-7.

**KS Statistic**  
Kolmogorov-Smirnov statistic — measures the maximum vertical distance between the cumulative distribution functions of default and non-default score distributions. The standard scorecard validation metric used by credit risk teams globally.

**MLflow Experiment Tracking**  
Logs every Optuna trial with its hyperparameters and AUC score. Allows comparing 30+ model configurations in a visual UI and reproducing any experiment exactly.

**Memory Optimization**  
Downcasting int64 → int32/int16 and float64 → float32 across all tables. Reduces memory footprint by ~50%, making it possible to process 57M+ rows on an 8GB machine.

---

## EDA Findings

- **8.1% default rate** — severe class imbalance requiring special handling
- **EXT_SOURCE_3** is the single strongest predictor (correlation -0.179 with target)
- **Males default at 10.1%** vs females at 7.0%
- **Unemployed applicants default at 36.4%** — highest risk income type
- **Lower secondary education** has the highest default rate (10.9%) among education levels
- **DAYS_EMPLOYED anomaly** — 55,374 rows with sentinel value 365,243 require cleaning
- **67 columns have missing values**, 41 above 50% — all building-level features
- **Younger applicants default more** — DAYS_BIRTH has +0.078 correlation with target

---

## What I Would Add With More Time

- **XGBoost comparison** — full side-by-side ROC curve comparison in the modeling notebook
- **Evidently AI drift monitoring** — automated data drift detection on new applicant batches
- **Model retraining pipeline** — trigger retraining when drift exceeds threshold
- **PostgreSQL feature store** — replace Parquet files with a proper feature store
- **CI/CD with GitHub Actions** — run tests and auto-deploy on every push
- **Authentication on the API** — JWT tokens for production use

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Data processing | Pandas, NumPy | Industry standard tabular data tools |
| Memory optimization | `pd.to_numeric(downcast=...)` | Handle 8GB RAM constraint with 57M rows |
| ML model | LightGBM | Best tabular performance, native NaN handling |
| Hyperparameter tuning | Optuna | Bayesian search, more efficient than GridSearch |
| Experiment tracking | MLflow | Reproducible experiments, standard in production |
| Explainability | SHAP TreeExplainer | Exact Shapley values, SR 11-7 compliant |
| API | FastAPI + Uvicorn | Async, auto-docs, production-grade Python API |
| Dashboard | Streamlit | Rapid ML dashboard prototyping |
| Containerization | Docker | Reproducible deployment across environments |
| Deployment | Render | Free tier, Docker-native, auto-deploy from GitHub |
| Visualization | Plotly | Interactive charts in Streamlit |

---

## Author

Built as a production-grade ML portfolio project targeting quantitative and data science roles in financial services.

**Interview one-liner:**  
*"I built a credit risk scoring system on Home Credit's 307k-application dataset, achieving 0.786 AUC and 0.489 KS statistic with LightGBM, implemented SHAP-based explainability aligned with SR 11-7 guidelines, and deployed it as a REST API with a real-time dashboard via Docker on Render."*