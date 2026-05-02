from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import load_model, predict

# ── App setup ──────────────────────────────────────────────
app = FastAPI(
    title="Loan Default Prediction API",
    description="Credit risk scoring API with SHAP explainability — SR 11-7 compliant",
    version="1.0.0"
)

# Load model once at startup
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
model = load_model(MODELS_DIR)
feature_names = model.booster_.feature_name()


# ── Request schema ──────────────────────────────────────────
class CustomerInput(BaseModel):
    # External credit scores — most predictive features
    EXT_SOURCE_1:           Optional[float] = None
    EXT_SOURCE_2:           Optional[float] = None
    EXT_SOURCE_3:           Optional[float] = None

    # Financial profile
    AMT_INCOME_TOTAL:       Optional[float] = None
    AMT_CREDIT:             Optional[float] = None
    AMT_ANNUITY:            Optional[float] = None
    AMT_GOODS_PRICE:        Optional[float] = None

    # Days features (negative = days before application)
    DAYS_BIRTH:             Optional[float] = None
    DAYS_EMPLOYED:          Optional[float] = None
    DAYS_REGISTRATION:      Optional[float] = None
    DAYS_ID_PUBLISH:        Optional[float] = None

    # Engineered features
    DAYS_EMPLOYED_ANOMALY:  Optional[float] = None
    AGE_YEARS:              Optional[float] = None
    EMPLOYED_YEARS:         Optional[float] = None
    CREDIT_INCOME_RATIO:    Optional[float] = None
    ANNUITY_INCOME_RATIO:   Optional[float] = None
    CREDIT_TERM:            Optional[float] = None
    GOODS_CREDIT_RATIO:     Optional[float] = None
    EXT_SOURCE_MEAN:        Optional[float] = None
    EXT_SOURCE_MIN:         Optional[float] = None
    EXT_SOURCE_STD:         Optional[float] = None
    INCOME_PER_PERSON:      Optional[float] = None

    # Demographics
    CNT_CHILDREN:           Optional[float] = None
    CNT_FAM_MEMBERS:        Optional[float] = None
    NAME_CONTRACT_TYPE:     Optional[str]   = None
    CODE_GENDER:            Optional[str]   = None
    NAME_INCOME_TYPE:       Optional[str]   = None
    NAME_EDUCATION_TYPE:    Optional[str]   = None
    NAME_FAMILY_STATUS:     Optional[str]   = None
    NAME_HOUSING_TYPE:      Optional[str]   = None
    FLAG_OWN_CAR:           Optional[str]   = None
    FLAG_OWN_REALTY:        Optional[str]   = None


# ── Endpoints ───────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Loan Default Prediction API",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs"
    }


@app.get("/health")
def health():
    return {
        "status":        "healthy",
        "model_loaded":  True,
        "n_features":    len(feature_names)
    }


@app.post("/predict")
def predict_default(customer: CustomerInput):
    """
    Predict default probability for a loan applicant.

    Returns:
    - risk_score: probability of default (0-1)
    - risk_percent: risk score as percentage
    - verdict: APPROVE / REVIEW / REJECT
    - explanations: top 5 SHAP factors driving the decision
    """
    try:
        customer_dict = customer.model_dump()
        result = predict(customer_dict, model, feature_names)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerInput]):
    """Predict for multiple customers at once. Max 100."""
    if len(customers) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 100 customers"
        )
    results = []
    for customer in customers:
        try:
            customer_dict = customer.model_dump()
            result = predict(customer_dict, model, feature_names)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})
    return results


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)