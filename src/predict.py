import pandas as pd
import numpy as np
import joblib
import shap
import os
import warnings
warnings.filterwarnings('ignore')


def load_model(models_dir: str):
    """Load the trained LightGBM model from disk."""
    model_path = os.path.join(models_dir, 'lgbm_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. "
                                f"Run src/train.py first.")
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def prepare_input(customer: dict, feature_names: list) -> pd.DataFrame:
    """
    Convert raw customer dict into a DataFrame that matches
    the exact feature set the model was trained on.
    Missing features are filled with NaN — model handles them natively.
    """
    df = pd.DataFrame([customer])

    # Add any missing columns as NaN
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan

    # Keep only columns the model knows about, in correct order
    df = df[feature_names]
    return df


def get_risk_score(model, input_df: pd.DataFrame) -> float:
    """
    Returns probability of default — a score between 0 and 1.
    Higher = more likely to default.
    """
    score = model.predict_proba(input_df)[:, 1][0]
    return float(score)


def get_risk_label(score: float) -> dict:
    """
    Convert raw score into human-readable verdict.
    Thresholds based on typical banking cut-offs.
    """
    if score < 0.3:
        return {
            'label':       'APPROVE',
            'color':       'green',
            'description': 'Low risk — recommend approval'
        }
    elif score < 0.5:
        return {
            'label':       'REVIEW',
            'color':       'orange',
            'description': 'Medium risk — manual review recommended'
        }
    else:
        return {
            'label':       'REJECT',
            'color':       'red',
            'description': 'High risk — recommend rejection'
        }


def get_shap_explanation(model, input_df: pd.DataFrame,
                         top_n: int = 5) -> list[dict]:
    """
    Generate SHAP values for a single prediction.
    Returns top N features driving the decision — both risk-increasing
    and risk-decreasing factors.
    SR 11-7 compliant explainability.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # For binary classification LightGBM returns list — take class 1
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    feature_names = input_df.columns.tolist()

    # Build explanation list
    explanations = []
    for i, (feat, val, shap_val) in enumerate(
        zip(feature_names, input_df.iloc[0], shap_vals)
    ):
        explanations.append({
            'feature':    feat,
            'value':      float(val) if not pd.isna(val) else None,
            'shap_value': float(shap_val),
            'direction':  'increases_risk' if shap_val > 0 else 'decreases_risk'
        })

    # Sort by absolute SHAP value — most impactful first
    explanations.sort(key=lambda x: abs(x['shap_value']), reverse=True)

    return explanations[:top_n]


def predict(customer: dict, model, feature_names: list) -> dict:
    """
    Master predict function.
    Input  — raw customer dict
    Output — risk score, verdict, top SHAP explanations
    """
    # Prepare input
    input_df = prepare_input(customer, feature_names)

    # Get risk score
    score = get_risk_score(model, input_df)

    # Get verdict
    verdict = get_risk_label(score)

    # Get SHAP explanation
    explanations = get_shap_explanation(model, input_df, top_n=5)

    return {
        'risk_score':   round(score, 4),
        'risk_percent': round(score * 100, 2),
        'verdict':      verdict,
        'explanations': explanations
    }


if __name__ == '__main__':
    import json

    MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

    # Load model
    model = load_model(MODELS_DIR)

    # Get feature names from model
    feature_names = model.booster_.feature_name()

    # Sample customer — typical low risk profile
    sample_customer = {
        'EXT_SOURCE_1':          0.65,
        'EXT_SOURCE_2':          0.70,
        'EXT_SOURCE_3':          0.68,
        'AMT_INCOME_TOTAL':      180000,
        'AMT_CREDIT':            450000,
        'AMT_ANNUITY':           22000,
        'AMT_GOODS_PRICE':       400000,
        'DAYS_BIRTH':            -14000,
        'DAYS_EMPLOYED':         -2500,
        'DAYS_EMPLOYED_ANOMALY': 0,
        'AGE_YEARS':             38.4,
        'EMPLOYED_YEARS':        6.8,
        'CREDIT_INCOME_RATIO':   2.5,
        'ANNUITY_INCOME_RATIO':  0.12,
        'EXT_SOURCE_MEAN':       0.676,
        'EXT_SOURCE_MIN':        0.65,
        'EXT_SOURCE_STD':        0.025,
    }

    result = predict(sample_customer, model, feature_names)

    print("\n=== PREDICTION RESULT ===")
    print(f"Risk Score:   {result['risk_score']}")
    print(f"Risk Percent: {result['risk_percent']}%")
    print(f"Verdict:      {result['verdict']['label']}")
    print(f"Description:  {result['verdict']['description']}")
    print(f"\nTop 5 SHAP Explanations:")
    for i, exp in enumerate(result['explanations'], 1):
        direction = "↑ risk" if exp['direction'] == 'increases_risk' else "↓ risk"
        print(f"  {i}. {exp['feature']:<35} "
              f"value={exp['value']}  "
              f"shap={exp['shap_value']:+.4f}  {direction}")

    print("\npredict.py working correctly")