"""BIM–LCA–Circularity AI — Main landing page."""
import streamlit as st
from pathlib import Path

from src.st_config import APP_TITLE, APP_ICON, APP_SUBTITLE, CSS_PATH, ASSETS_DIR

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ──────────────────────────────────────────────────────────
# Load Material Symbols font for sidebar toggle icon
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet">',
    unsafe_allow_html=True,
)
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### {APP_ICON} {APP_TITLE}")
    st.markdown(
        '<p style="color: #8892A4; font-size: 0.8rem;">'
        "AI-driven sustainability & circularity assessment for construction projects"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Hero Section ──────────────────────────────────────────────────────
st.markdown(
    """
    <div style="padding: 1rem 0 0.5rem;">
        <span class="hero-title">BIM–LCA–Circularity AI</span>
        <p class="hero-subtitle">
            An AI-based predictive model leveraging BIM-use data across the building lifecycle
            for multicriteria sustainability assessment — targeting decarbonisation and building
            circularity — embedded in a BIM–AI–MCDM digital workflow.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────
import json

eda_path = ASSETS_DIR / "eda_stats.json"
model_path = ASSETS_DIR / "model_results.json"

n_obs = 199
best_f1 = "1.00"
n_phases = 6

if eda_path.exists():
    with open(eda_path) as f:
        eda = json.load(f)
    n_obs = eda.get("n_rows", 199)

if model_path.exists():
    with open(model_path) as f:
        models = json.load(f)
    # Find best non-baseline macro_f1
    f1_values = [
        r.get("macro_f1_mean", 0) for r in models
        if r.get("model") and "Baseline" not in str(r.get("model", ""))
        and r.get("macro_f1_mean") is not None
    ]
    if f1_values:
        best_f1 = f"{max(f1_values):.2f}"

cols = st.columns(3)
kpi_data = [
    (str(n_obs), "Observations Analysed", "3 countries"),
    (best_f1, "Best Macro-F1 Score", "Random Forest"),
    (str(n_phases), "BIM Lifecycle Phases", "Full coverage"),
]

for col, (value, label, delta) in zip(cols, kpi_data):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-delta-up">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Methodology Brief ────────────────────────────────────────────────
st.markdown(
    """
    <div class="callout">
        <strong>Methodology.</strong>
        This tool integrates Building Information Modelling (BIM) adoption data with machine learning
        to predict sustainability outcomes (waste reduction, CO₂ emissions, recycling rates) and rank
        construction scenarios using Multi-Criteria Decision Making (TOPSIS). Survey data from
        Slovakia, Croatia, and Slovenia was validated via EFA/CFA, modelled with ordinal regression
        and PLS-SEM, then fed into Random Forest, XGBoost, LightGBM, and SVM classifiers with
        SHAP-based interpretability.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── CTA ───────────────────────────────────────────────────────────────
st.markdown("")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="cta-button">', unsafe_allow_html=True)
    if st.button("Try the Predictor  →", use_container_width=True):
        st.switch_page("pages/2_🤖_Predictor.py")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        <p>
            Research: AI-Based Predictive Integrated Model for BIM–LCA–Circularity Assessment<br>
            Target: Q1/Q2 Scopus Journal (Journal of Cleaner Production / Automation in Construction)<br>
            <br>
            Built with <a href="https://streamlit.io" target="_blank">Streamlit</a> &middot;
            Models: scikit-learn, XGBoost, LightGBM &middot;
            Interpretability: SHAP
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
