import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

# ── Header ─────────────────────────────────────────────────
st.title("🏦 Loan Default Prediction System")
st.markdown("Credit risk scoring with AI explainability — SR 11-7 compliant")
st.divider()

# ── Sidebar — customer input form ──────────────────────────
st.sidebar.header("Customer Profile")

with st.sidebar:
    st.subheader("External Credit Scores")
    ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5, 0.01)
    ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.5, 0.01)
    ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.5, 0.01)

    st.subheader("Financial Profile")
    income       = st.number_input("Annual Income (₹)",
                                   min_value=0, value=180000, step=10000)
    credit       = st.number_input("Loan Amount (₹)",
                                   min_value=0, value=450000, step=10000)
    annuity      = st.number_input("Monthly Annuity (₹)",
                                   min_value=0, value=22000, step=1000)
    goods_price  = st.number_input("Goods Price (₹)",
                                   min_value=0, value=400000, step=10000)

    st.subheader("Personal Details")
    age          = st.slider("Age (years)", 18, 70, 35)
    employed     = st.slider("Years Employed", 0, 40, 5)
    children     = st.number_input("Number of Children", 0, 10, 0)

    st.subheader("Demographics")
    gender       = st.selectbox("Gender", ["M", "F"])
    education    = st.selectbox("Education", [
                    "Higher education",
                    "Secondary / secondary special",
                    "Incomplete higher",
                    "Lower secondary",
                    "Academic degree"
                  ])
    income_type  = st.selectbox("Income Type", [
                    "Working", "Commercial associate",
                    "State servant", "Pensioner"
                  ])
    owns_car     = st.selectbox("Owns Car", ["Y", "N"])
    owns_realty  = st.selectbox("Owns Realty", ["Y", "N"])

    predict_btn  = st.button("🔍 Predict Risk", use_container_width=True)


# ── Helper functions ────────────────────────────────────────
def build_gauge(score: float, label: str, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        number={'suffix': "%", 'font': {'size': 48}},
        delta={'reference': 30, 'increasing': {'color': "red"},
               'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar':  {'color': color, 'thickness': 0.3},
            'steps': [
                {'range': [0,  30], 'color': '#d4edda'},
                {'range': [30, 50], 'color': '#fff3cd'},
                {'range': [50, 100],'color': '#f8d7da'},
            ],
            'threshold': {
                'line':  {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score * 100
            }
        },
        title={'text': f"Default Risk Score<br><b>{label}</b>",
               'font': {'size': 20}}
    ))
    fig.update_layout(height=350, margin=dict(t=80, b=0, l=40, r=40))
    return fig


def build_shap_chart(explanations: list):
    features   = [e['feature']    for e in explanations]
    shap_vals  = [e['shap_value'] for e in explanations]
    colors     = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_vals]
    directions = ['↑ Increases risk' if v > 0
                  else '↓ Decreases risk' for v in shap_vals]

    fig = go.Figure(go.Bar(
        x=shap_vals,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{d}  ({v:+.3f})"
              for d, v in zip(directions, shap_vals)],
        textposition='outside',
    ))
    fig.update_layout(
        title="Top 5 Factors Driving This Decision (SHAP)",
        xaxis_title="SHAP Value (impact on risk score)",
        height=350,
        margin=dict(t=60, b=40, l=200, r=200),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ── Main panel ─────────────────────────────────────────────
if predict_btn:
    # Build payload
    payload = {
        "EXT_SOURCE_1":         ext1,
        "EXT_SOURCE_2":         ext2,
        "EXT_SOURCE_3":         ext3,
        "AMT_INCOME_TOTAL":     float(income),
        "AMT_CREDIT":           float(credit),
        "AMT_ANNUITY":          float(annuity),
        "AMT_GOODS_PRICE":      float(goods_price),
        "DAYS_BIRTH":           float(-age * 365),
        "DAYS_EMPLOYED":        float(-employed * 365),
        "DAYS_EMPLOYED_ANOMALY":0,
        "AGE_YEARS":            float(age),
        "EMPLOYED_YEARS":       float(employed),
        "CREDIT_INCOME_RATIO":  float(credit / (income + 1)),
        "ANNUITY_INCOME_RATIO": float(annuity / (income + 1)),
        "CREDIT_TERM":          float(annuity / (credit + 1)),
        "GOODS_CREDIT_RATIO":   float(goods_price / (credit + 1)),
        "EXT_SOURCE_MEAN":      float(np.mean([ext1, ext2, ext3])),
        "EXT_SOURCE_MIN":       float(np.min([ext1, ext2, ext3])),
        "EXT_SOURCE_STD":       float(np.std([ext1, ext2, ext3])),
        "CNT_CHILDREN":         float(children),
        "CODE_GENDER":          gender,
        "NAME_EDUCATION_TYPE":  education,
        "NAME_INCOME_TYPE":     income_type,
        "FLAG_OWN_CAR":         owns_car,
        "FLAG_OWN_REALTY":      owns_realty,
    }

    with st.spinner("Calculating risk score..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )
            result = response.json()

            score   = result['risk_score']
            percent = result['risk_percent']
            verdict = result['verdict']
            explanations = result['explanations']

            # ── Results layout ──────────────────────────────
            col1, col2 = st.columns([1, 1])

            with col1:
                color_map = {
                    'green':  '#2ecc71',
                    'orange': '#f39c12',
                    'red':    '#e74c3c'
                }
                fig_gauge = build_gauge(
                    score,
                    verdict['label'],
                    color_map[verdict['color']]
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col2:
                st.subheader("Decision Summary")
                color = verdict['color']
                label = verdict['label']

                if label == 'APPROVE':
                    st.success(f"✅ {label} — {verdict['description']}")
                elif label == 'REVIEW':
                    st.warning(f"⚠️ {label} — {verdict['description']}")
                else:
                    st.error(f"❌ {label} — {verdict['description']}")

                st.metric("Default Probability", f"{percent}%")
                st.metric("EXT_SOURCE Average",
                          f"{np.mean([ext1,ext2,ext3]):.3f}")
                st.metric("Credit / Income Ratio",
                          f"{credit/(income+1):.2f}x")

                st.subheader("Input Summary")
                summary_df = pd.DataFrame({
                    'Feature': ['Age', 'Income', 'Loan Amount',
                                'Employment', 'Education'],
                    'Value':   [f"{age} years",
                                f"₹{income:,}",
                                f"₹{credit:,}",
                                f"{employed} years",
                                education]
                })
                st.dataframe(summary_df, hide_index=True,
                             use_container_width=True)

            # ── SHAP chart ──────────────────────────────────
            st.divider()
            fig_shap = build_shap_chart(explanations)
            st.plotly_chart(fig_shap, use_container_width=True)

            # ── Raw explanation table ───────────────────────
            st.subheader("Detailed SHAP Breakdown")
            exp_df = pd.DataFrame(explanations)
            exp_df['shap_value'] = exp_df['shap_value'].round(4)
            st.dataframe(exp_df, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"API Error: {e}")
            st.info("Make sure the FastAPI server is running on port 8000")

else:
    # Default state — show instructions
    st.info("👈 Fill in the customer profile in the sidebar and click "
            "**Predict Risk** to get a decision.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "LightGBM")
    with col2:
        st.metric("AUC Score", "0.786")
    with col3:
        st.metric("KS Statistic", "0.489")

    st.markdown("""
    ### How this works
    1. Fill in the customer's financial profile in the sidebar
    2. Click **Predict Risk**
    3. The model returns a default probability score
    4. SHAP values explain exactly why the model made that decision
    
    ### Model Info
    - Trained on 307,511 loan applications from Home Credit
    - 210 engineered features from 7 data tables
    - LightGBM with Optuna hyperparameter tuning
    - Explainability aligned with SR 11-7 model risk guidelines
    """)