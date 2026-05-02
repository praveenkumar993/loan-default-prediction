import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="CreditIQ — Loan Risk Engine",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Root theme */
:root {
    --bg:        #080c14;
    --surface:   #0d1420;
    --surface2:  #111927;
    --border:    #1e2d42;
    --accent:    #00d4ff;
    --accent2:   #0099bb;
    --green:     #00e5a0;
    --amber:     #ffb347;
    --red:       #ff4d6d;
    --text:      #e2eaf4;
    --muted:     #5a7a9a;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

/* Full app background */
.stApp {
    background: var(--bg) !important;
    font-family: var(--font-body) !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    font-family: var(--font-body) !important;
    color: var(--text) !important;
}

/* All text */
.stMarkdown, .stText, p, span, label, div {
    font-family: var(--font-body) !important;
    color: var(--text) !important;
}

/* Sliders */
[data-testid="stSlider"] > div > div > div {
    background: var(--accent) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 12px rgba(0,212,255,0.5) !important;
}

/* Number inputs */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
}

/* Select boxes */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}

/* Button */
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #00d4ff 0%, #0066ff 100%) !important;
    color: #000 !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 12px 24px !important;
    box-shadow: 0 0 24px rgba(0,212,255,0.25) !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] button:hover {
    box-shadow: 0 0 36px rgba(0,212,255,0.5) !important;
    transform: translateY(-1px) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-head) !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Success/Warning/Error */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
    background: var(--surface2) !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* Sidebar section labels */
.sidebar-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent) !important;
    margin: 20px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)


# ── Top header bar ──────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(90deg, #0d1420 0%, #0a1628 50%, #0d1420 100%);
    border-bottom: 1px solid #1e2d42;
    padding: 20px 32px;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
">
    <div style="display:flex; align-items:center; gap:14px;">
        <div style="
            width:38px; height:38px;
            background: linear-gradient(135deg,#00d4ff,#0066ff);
            clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
            flex-shrink:0;
        "></div>
        <div>
            <div style="font-family:'Syne',sans-serif; font-size:20px; font-weight:800; color:#e2eaf4; letter-spacing:-0.5px;">
                CreditIQ
            </div>
            <div style="font-family:'DM Mono',monospace; font-size:10px; color:#5a7a9a; letter-spacing:2px; text-transform:uppercase; margin-top:1px;">
                Risk Intelligence Engine
            </div>
        </div>
    </div>
    <div style="display:flex; gap:24px; align-items:center;">
        <div style="text-align:center;">
            <div style="font-family:'DM Mono',monospace; font-size:10px; color:#5a7a9a; letter-spacing:1px; text-transform:uppercase;">Model</div>
            <div style="font-family:'Syne',sans-serif; font-size:13px; font-weight:600; color:#00d4ff;">LightGBM</div>
        </div>
        <div style="width:1px; height:28px; background:#1e2d42;"></div>
        <div style="text-align:center;">
            <div style="font-family:'DM Mono',monospace; font-size:10px; color:#5a7a9a; letter-spacing:1px; text-transform:uppercase;">AUC</div>
            <div style="font-family:'Syne',sans-serif; font-size:13px; font-weight:600; color:#00e5a0;">0.786</div>
        </div>
        <div style="width:1px; height:28px; background:#1e2d42;"></div>
        <div style="text-align:center;">
            <div style="font-family:'DM Mono',monospace; font-size:10px; color:#5a7a9a; letter-spacing:1px; text-transform:uppercase;">KS Stat</div>
            <div style="font-family:'Syne',sans-serif; font-size:13px; font-weight:600; color:#00e5a0;">0.489</div>
        </div>
        <div style="width:1px; height:28px; background:#1e2d42;"></div>
        <div style="text-align:center;">
            <div style="font-family:'DM Mono',monospace; font-size:10px; color:#5a7a9a; letter-spacing:1px; text-transform:uppercase;">Training Set</div>
            <div style="font-family:'Syne',sans-serif; font-size:13px; font-weight:600; color:#e2eaf4;">307,511</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-label">⬡ Applicant Profile</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Credit Bureau Scores</div>', unsafe_allow_html=True)
    ext1 = st.slider("EXT SOURCE 1", 0.0, 1.0, 0.50, 0.01, help="External credit bureau score 1")
    ext2 = st.slider("EXT SOURCE 2", 0.0, 1.0, 0.55, 0.01, help="External credit bureau score 2")
    ext3 = st.slider("EXT SOURCE 3", 0.0, 1.0, 0.52, 0.01, help="External credit bureau score 3")

    ext_avg = np.mean([ext1, ext2, ext3])
    bar_color = "#00e5a0" if ext_avg > 0.5 else "#ffb347" if ext_avg > 0.35 else "#ff4d6d"
    st.markdown(f"""
    <div style="background:#111927; border:1px solid #1e2d42; border-radius:8px; padding:12px; margin:8px 0;">
        <div style="font-family:'DM Mono',monospace; font-size:10px; color:#5a7a9a; letter-spacing:1px; text-transform:uppercase; margin-bottom:6px;">Bureau Score Average</div>
        <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:700; color:{bar_color};">{ext_avg:.3f}</div>
        <div style="background:#1e2d42; border-radius:4px; height:4px; margin-top:8px;">
            <div style="background:{bar_color}; width:{ext_avg*100:.0f}%; height:4px; border-radius:4px; box-shadow:0 0 8px {bar_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Financials</div>', unsafe_allow_html=True)
    income      = st.number_input("Annual Income (₹)", min_value=0, value=180000, step=10000)
    credit      = st.number_input("Loan Amount (₹)",   min_value=0, value=450000, step=10000)
    annuity     = st.number_input("Monthly Annuity (₹)",min_value=0, value=22000,  step=1000)
    goods_price = st.number_input("Goods Price (₹)",   min_value=0, value=400000, step=10000)

    ratio = credit / (income + 1)
    ratio_color = "#00e5a0" if ratio < 3 else "#ffb347" if ratio < 6 else "#ff4d6d"
    st.markdown(f"""
    <div style="background:#111927; border:1px solid #1e2d42; border-radius:8px; padding:12px; margin:8px 0;">
        <div style="font-family:'DM Mono',monospace; font-size:10px; color:#5a7a9a; letter-spacing:1px; text-transform:uppercase; margin-bottom:4px;">Credit / Income Ratio</div>
        <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:700; color:{ratio_color};">{ratio:.2f}x</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Personal</div>', unsafe_allow_html=True)
    age      = st.slider("Age (years)",       18, 70, 35)
    employed = st.slider("Years Employed",     0, 40,  5)
    children = st.number_input("No. of Children", 0, 10, 0)

    st.markdown('<div class="sidebar-label">Demographics</div>', unsafe_allow_html=True)
    gender      = st.selectbox("Gender",       ["M", "F"])
    education   = st.selectbox("Education",    [
                    "Higher education",
                    "Secondary / secondary special",
                    "Incomplete higher",
                    "Lower secondary",
                    "Academic degree"])
    income_type = st.selectbox("Income Type",  [
                    "Working", "Commercial associate",
                    "State servant", "Pensioner"])
    owns_car    = st.selectbox("Owns Car",     ["Y", "N"])
    owns_realty = st.selectbox("Owns Realty",  ["Y", "N"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⬡  RUN RISK ANALYSIS", use_container_width=True)


# ── Helper — Gauge ───────────────────────────────────────────────────────────
def build_gauge(score: float, label: str):
    pct = score * 100
    if score < 0.3:
        bar_color, glow = "#00e5a0", "rgba(0,229,160,0.3)"
    elif score < 0.5:
        bar_color, glow = "#ffb347", "rgba(255,179,71,0.3)"
    else:
        bar_color, glow = "#ff4d6d", "rgba(255,77,109,0.3)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={
            'suffix': "%",
            'font': {'size': 56, 'color': bar_color,
                     'family': 'Syne, sans-serif'},
            'valueformat': '.1f'
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 0,
                'tickcolor': "#1e2d42",
                'tickvals': [0, 30, 50, 100],
                'ticktext': ['0', '30', '50', '100'],
                'tickfont': {'color': '#5a7a9a', 'size': 11,
                             'family': 'DM Mono'}
            },
            'bar':   {'color': bar_color, 'thickness': 0.28},
            'bgcolor': '#0d1420',
            'borderwidth': 0,
            'steps': [
                {'range': [0,  30], 'color': 'rgba(0,229,160,0.06)'},
                {'range': [30, 50], 'color': 'rgba(255,179,71,0.06)'},
                {'range': [50,100], 'color': 'rgba(255,77,109,0.06)'},
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 3},
                'thickness': 0.85,
                'value': pct
            }
        },
        title={
            'text': f"<b style='font-family:Syne'>{label}</b><br>"
                    f"<span style='font-size:12px;color:#5a7a9a;"
                    f"font-family:DM Mono'>DEFAULT PROBABILITY</span>",
            'font': {'size': 18, 'color': bar_color,
                     'family': 'Syne, sans-serif'}
        }
    ))
    fig.update_layout(
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=90, b=10, l=30, r=30),
        font_color='#e2eaf4'
    )
    return fig


# ── Helper — SHAP chart ──────────────────────────────────────────────────────
def build_shap_chart(explanations: list):
    features  = [e['feature']    for e in explanations]
    shap_vals = [e['shap_value'] for e in explanations]
    colors    = ['#ff4d6d' if v > 0 else '#00e5a0' for v in shap_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=shap_vals,
        y=features,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        text=[f"{'+' if v>0 else ''}{v:.4f}" for v in shap_vals],
        textposition='outside',
        textfont=dict(family='DM Mono', size=11, color='#5a7a9a'),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "SHAP: %{x:.4f}<br>"
            "<extra></extra>"
        )
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=280,
        margin=dict(t=20, b=20, l=220, r=120),
        xaxis=dict(
            showgrid=True,
            gridcolor='#1e2d42',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='#2a3d54',
            zerolinewidth=2,
            tickfont=dict(family='DM Mono', size=10, color='#5a7a9a'),
            title=dict(text='SHAP Impact Value',
                       font=dict(family='DM Mono', size=10,
                                 color='#5a7a9a'))
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(family='DM Mono', size=11, color='#e2eaf4')
        ),
        bargap=0.35,
    )
    return fig


# ── Card helper ──────────────────────────────────────────────────────────────
def card(title, value, subtitle="", color="#00d4ff"):
    return f"""
    <div style="
        background: #0d1420;
        border: 1px solid #1e2d42;
        border-top: 2px solid {color};
        border-radius: 10px;
        padding: 18px 20px;
        height: 100%;
    ">
        <div style="font-family:'DM Mono',monospace; font-size:10px;
                    color:#5a7a9a; letter-spacing:1.5px;
                    text-transform:uppercase; margin-bottom:8px;">{title}</div>
        <div style="font-family:'Syne',sans-serif; font-size:26px;
                    font-weight:700; color:{color}; line-height:1.1;">{value}</div>
        <div style="font-family:'DM Sans',sans-serif; font-size:12px;
                    color:#5a7a9a; margin-top:4px;">{subtitle}</div>
    </div>
    """


# ── Main panel ───────────────────────────────────────────────────────────────
if predict_btn:
    payload = {
        "EXT_SOURCE_1":          ext1,
        "EXT_SOURCE_2":          ext2,
        "EXT_SOURCE_3":          ext3,
        "AMT_INCOME_TOTAL":      float(income),
        "AMT_CREDIT":            float(credit),
        "AMT_ANNUITY":           float(annuity),
        "AMT_GOODS_PRICE":       float(goods_price),
        "DAYS_BIRTH":            float(-age * 365),
        "DAYS_EMPLOYED":         float(-employed * 365),
        "DAYS_EMPLOYED_ANOMALY": 0,
        "AGE_YEARS":             float(age),
        "EMPLOYED_YEARS":        float(employed),
        "CREDIT_INCOME_RATIO":   float(credit / (income + 1)),
        "ANNUITY_INCOME_RATIO":  float(annuity / (income + 1)),
        "CREDIT_TERM":           float(annuity / (credit + 1)),
        "GOODS_CREDIT_RATIO":    float(goods_price / (credit + 1)),
        "EXT_SOURCE_MEAN":       float(np.mean([ext1, ext2, ext3])),
        "EXT_SOURCE_MIN":        float(np.min([ext1, ext2, ext3])),
        "EXT_SOURCE_STD":        float(np.std([ext1, ext2, ext3])),
        "CNT_CHILDREN":          float(children),
        "CODE_GENDER":           gender,
        "NAME_EDUCATION_TYPE":   education,
        "NAME_INCOME_TYPE":      income_type,
        "FLAG_OWN_CAR":          owns_car,
        "FLAG_OWN_REALTY":       owns_realty,
    }

    with st.spinner("Running risk analysis..."):
        try:
            response = requests.post(f"{API_URL}/predict",
                                     json=payload, timeout=30)
            result   = response.json()

            score        = result['risk_score']
            percent      = result['risk_percent']
            verdict      = result['verdict']
            explanations = result['explanations']
            label        = verdict['label']

            # ── Verdict banner ──────────────────────────────────────────────
            v_color = {"APPROVE":"#00e5a0",
                       "REVIEW": "#ffb347",
                       "REJECT": "#ff4d6d"}[label]
            v_icon  = {"APPROVE":"✦", "REVIEW":"◈", "REJECT":"✕"}[label]
            v_bg    = {"APPROVE":"rgba(0,229,160,0.07)",
                       "REVIEW": "rgba(255,179,71,0.07)",
                       "REJECT": "rgba(255,77,109,0.07)"}[label]

            st.markdown(f"""
            <div style="
                background:{v_bg};
                border:1px solid {v_color}33;
                border-left:4px solid {v_color};
                border-radius:10px;
                padding:16px 24px;
                display:flex;
                align-items:center;
                justify-content:space-between;
                margin-bottom:24px;
            ">
                <div style="display:flex; align-items:center; gap:14px;">
                    <div style="font-size:24px; color:{v_color};">{v_icon}</div>
                    <div>
                        <div style="font-family:'Syne',sans-serif; font-size:18px;
                                    font-weight:700; color:{v_color};">{label}</div>
                        <div style="font-family:'DM Sans',sans-serif; font-size:13px;
                                    color:#5a7a9a; margin-top:2px;">{verdict['description']}</div>
                    </div>
                </div>
                <div style="font-family:'DM Mono',monospace; font-size:32px;
                            font-weight:500; color:{v_color};">
                    {percent}%
                    <span style="font-size:12px; color:#5a7a9a; display:block;
                                 text-align:right; margin-top:2px;">default risk</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Gauge + metrics row ─────────────────────────────────────────
            col_g, col_m1, col_m2, col_m3 = st.columns([2, 1, 1, 1])

            with col_g:
                st.markdown(f"""
                <div style="background:#0d1420; border:1px solid #1e2d42;
                            border-radius:12px; padding:8px;">
                """, unsafe_allow_html=True)
                fig_gauge = build_gauge(score, label)
                st.plotly_chart(fig_gauge, use_container_width=True,
                                config={'displayModeBar': False})
                st.markdown("</div>", unsafe_allow_html=True)

            with col_m1:
                st.markdown(card(
                    "Bureau Score Avg",
                    f"{np.mean([ext1,ext2,ext3]):.3f}",
                    "Higher = safer",
                    "#00d4ff"
                ), unsafe_allow_html=True)

            with col_m2:
                r = credit / (income + 1)
                rc = "#00e5a0" if r < 3 else "#ffb347" if r < 6 else "#ff4d6d"
                st.markdown(card(
                    "Credit / Income",
                    f"{r:.2f}x",
                    "Lower = safer",
                    rc
                ), unsafe_allow_html=True)

            with col_m3:
                ann_r = annuity / (income + 1)
                ac = "#00e5a0" if ann_r < 0.15 else "#ffb347" if ann_r < 0.3 else "#ff4d6d"
                st.markdown(card(
                    "Annuity / Income",
                    f"{ann_r:.3f}",
                    "Lower = safer",
                    ac
                ), unsafe_allow_html=True)

            # ── SHAP section ────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:'DM Mono',monospace; font-size:10px;
                        letter-spacing:2px; text-transform:uppercase;
                        color:#5a7a9a; margin-bottom:12px;">
                ⬡ &nbsp; Model Explainability — SHAP Feature Impact
            </div>
            """, unsafe_allow_html=True)

            col_shap, col_table = st.columns([3, 2])

            with col_shap:
                st.markdown("""
                <div style="background:#0d1420; border:1px solid #1e2d42;
                            border-radius:12px; padding:16px;">
                """, unsafe_allow_html=True)
                fig_shap = build_shap_chart(explanations)
                st.plotly_chart(fig_shap, use_container_width=True,
                                config={'displayModeBar': False})
                st.markdown("</div>", unsafe_allow_html=True)

            with col_table:
                st.markdown("""
                <div style="background:#0d1420; border:1px solid #1e2d42;
                            border-radius:12px; padding:20px;">
                    <div style="font-family:'DM Mono',monospace; font-size:10px;
                                letter-spacing:1.5px; text-transform:uppercase;
                                color:#5a7a9a; margin-bottom:14px;">
                        Top Risk Drivers
                    </div>
                """, unsafe_allow_html=True)

                for i, exp in enumerate(explanations):
                    sv   = exp['shap_value']
                    fv   = f"{exp['value']:.4f}" if exp['value'] else "—"
                    dc   = "#ff4d6d" if sv > 0 else "#00e5a0"
                    darr = "▲" if sv > 0 else "▼"
                    dlbl = "Risk +" if sv > 0 else "Risk −"
                    st.markdown(f"""
                    <div style="
                        display:flex; align-items:center; justify-content:space-between;
                        padding:10px 0;
                        border-bottom: 1px solid #1e2d42;
                    ">
                        <div>
                            <div style="font-family:'DM Mono',monospace; font-size:11px;
                                        color:#e2eaf4;">{exp['feature']}</div>
                            <div style="font-family:'DM Mono',monospace; font-size:10px;
                                        color:#5a7a9a; margin-top:2px;">val: {fv}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-family:'Syne',sans-serif; font-size:14px;
                                        font-weight:700; color:{dc};">
                                {darr} {abs(sv):.4f}
                            </div>
                            <div style="font-family:'DM Mono',monospace; font-size:9px;
                                        color:{dc}; letter-spacing:1px;">{dlbl}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            # ── Applicant summary ───────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:'DM Mono',monospace; font-size:10px;
                        letter-spacing:2px; text-transform:uppercase;
                        color:#5a7a9a; margin-bottom:12px;">
                ⬡ &nbsp; Applicant Profile Summary
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(6)
            profile = [
                ("Age",        f"{age} yrs",      "#00d4ff"),
                ("Income",     f"₹{income:,}",    "#00e5a0"),
                ("Loan",       f"₹{credit:,}",    "#00d4ff"),
                ("Employed",   f"{employed} yrs",  "#00e5a0"),
                ("Education",  education[:12]+"…" if len(education)>12
                               else education,    "#00d4ff"),
                ("Gender",     gender,             "#00e5a0"),
            ]
            for col, (ttl, val, clr) in zip(cols, profile):
                with col:
                    st.markdown(card(ttl, val, color=clr),
                                unsafe_allow_html=True)

        except Exception as e:
            st.error(f"API Error: {e}")
            st.info("Ensure the FastAPI server is running → "
                    "`python -m uvicorn api.main:app --port 8000`")

else:
    # ── Landing state ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        text-align:center;
        padding: 60px 20px;
        max-width: 700px;
        margin: 0 auto;
    ">
        <div style="
            width:80px; height:80px;
            background: linear-gradient(135deg,#00d4ff11,#0066ff22);
            border: 1px solid #1e2d42;
            clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
            margin: 0 auto 28px;
            display:flex; align-items:center; justify-content:center;
        ">
            <div style="
                width:40px; height:40px;
                background: linear-gradient(135deg,#00d4ff,#0066ff);
                clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
            "></div>
        </div>
        <div style="font-family:'Syne',sans-serif; font-size:36px; font-weight:800;
                    color:#e2eaf4; line-height:1.1; margin-bottom:16px;">
            AI-Powered Credit Risk
        </div>
        <div style="font-family:'DM Sans',sans-serif; font-size:15px; color:#5a7a9a;
                    line-height:1.7; margin-bottom:40px;">
            Enter an applicant's financial profile in the sidebar.<br>
            The engine returns a real-time default probability with<br>
            SHAP-based explainability aligned to SR 11-7 guidelines.
        </div>
        <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
    """, unsafe_allow_html=True)

    stat_cols = st.columns(4)
    stats = [
        ("307,511",  "Training samples",  "#00d4ff"),
        ("210",      "Engineered features","#00e5a0"),
        ("0.786",    "AUC-ROC score",     "#00d4ff"),
        ("0.489",    "KS Statistic",      "#00e5a0"),
    ]
    for col, (val, lbl, clr) in zip(stat_cols, stats):
        with col:
            st.markdown(f"""
            <div style="background:#0d1420; border:1px solid #1e2d42;
                        border-top:2px solid {clr}; border-radius:10px;
                        padding:20px 16px; text-align:center;">
                <div style="font-family:'Syne',sans-serif; font-size:28px;
                            font-weight:800; color:{clr};">{val}</div>
                <div style="font-family:'DM Mono',monospace; font-size:10px;
                            color:#5a7a9a; text-transform:uppercase;
                            letter-spacing:1px; margin-top:6px;">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0d1420; border:1px solid #1e2d42; border-radius:12px;
                padding:28px 32px; max-width:700px; margin:0 auto;">
        <div style="font-family:'DM Mono',monospace; font-size:10px; color:#5a7a9a;
                    letter-spacing:2px; text-transform:uppercase; margin-bottom:20px;">
            System Architecture
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
            <div style="padding:14px; background:#111927; border-radius:8px;
                        border-left:3px solid #00d4ff;">
                <div style="font-family:'Syne',sans-serif; font-size:13px;
                            font-weight:600; color:#e2eaf4; margin-bottom:4px;">
                    Data Pipeline
                </div>
                <div style="font-family:'DM Sans',sans-serif; font-size:12px; color:#5a7a9a;">
                    7 CSV tables · 27M+ rows aggregated per customer
                </div>
            </div>
            <div style="padding:14px; background:#111927; border-radius:8px;
                        border-left:3px solid #00e5a0;">
                <div style="font-family:'Syne',sans-serif; font-size:13px;
                            font-weight:600; color:#e2eaf4; margin-bottom:4px;">
                    Model
                </div>
                <div style="font-family:'DM Sans',sans-serif; font-size:12px; color:#5a7a9a;">
                    LightGBM · Optuna tuning · 30 trials
                </div>
            </div>
            <div style="padding:14px; background:#111927; border-radius:8px;
                        border-left:3px solid #00d4ff;">
                <div style="font-family:'Syne',sans-serif; font-size:13px;
                            font-weight:600; color:#e2eaf4; margin-bottom:4px;">
                    Explainability
                </div>
                <div style="font-family:'DM Sans',sans-serif; font-size:12px; color:#5a7a9a;">
                    SHAP TreeExplainer · SR 11-7 compliant
                </div>
            </div>
            <div style="padding:14px; background:#111927; border-radius:8px;
                        border-left:3px solid #00e5a0;">
                <div style="font-family:'Syne',sans-serif; font-size:13px;
                            font-weight:600; color:#e2eaf4; margin-bottom:4px;">
                    Deployment
                </div>
                <div style="font-family:'DM Sans',sans-serif; font-size:12px; color:#5a7a9a;">
                    FastAPI · Docker · Render
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)