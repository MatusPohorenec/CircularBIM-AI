"""📄 About — Methodology, citation, and project information."""
import streamlit as st

from src.st_config import APP_TITLE, APP_ICON, CSS_PATH, C_ACCENT, C_MUTED

st.set_page_config(page_title=f"{APP_TITLE} — About", page_icon="📄", layout="wide")

st.markdown('<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet">', unsafe_allow_html=True)
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"### {APP_ICON} {APP_TITLE}")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("## 📄 About This Project")
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Paper Information ─────────────────────────────────────────────────
st.markdown("### Research Paper")
st.markdown(
    '<div class="callout">'
    "<strong>Title:</strong> AI-Based Predictive Integrated Model for BIM–LCA–Circularity "
    "Assessment in Construction Projects<br><br>"
    "<strong>Target Journal:</strong> Q1/Q2 Scopus Journal (Journal of Cleaner Production / "
    "Automation in Construction)<br><br>"
    "<strong>Status:</strong> Manuscript in preparation"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Methodology ───────────────────────────────────────────────────────
st.markdown("### Methodology")

st.markdown(
    """
The research follows a six-phase pipeline integrating survey-based BIM adoption data with 
machine learning for sustainability assessment:

**Phase 1 — Data Collection & EDA**  
Survey data from 199 construction industry respondents across Slovakia, Croatia, and Slovenia. 
Variables cover BIM usage across 6 building lifecycle phases and 3 sustainability outcome 
indicators (recycling rate, waste reduction, CO₂ reduction).

**Phase 2 — Construct Validation**  
Internal consistency via Cronbach's α and McDonald's ω. Exploratory Factor Analysis (EFA) with 
varimax rotation, Confirmatory Factor Analysis (CFA) using semopy, Average Variance Extracted (AVE), 
and Heterotrait-Monotrait ratio (HTMT) for discriminant validity.

**Phase 3 — Inferential Modelling**  
Ordinal Logistic Regression for each sustainability target to identify statistically significant 
BIM predictors. PLS-SEM path model: BIM Maturity → Sustainability Impact with project/company 
size as controls.

**Phase 4 — Predictive AI Modelling**  
Ordinal classification using Random Forest, XGBoost, LightGBM, SVM, and mord ordinal regressors. 
5×5 repeated stratified k-fold cross-validation. SHAP-based feature importance and dependence analysis.

**Phase 5 — MCDM Scenario Analysis**  
Six BIM adoption scenarios evaluated via TOPSIS with equal, entropy-based, and AHP-derived weights. 
Sensitivity analysis (±20% weight variation) validates ranking stability.

**Phase 6 — Validation & Self-Challenge**  
Post-hoc power analysis, overfitting diagnostics, data leakage checks, country-level performance 
breakdown, and consolidated limitation reporting.
"""
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Methodology Flow Diagram ─────────────────────────────────────────
st.markdown("### Research Pipeline")

flow_html = """
<div style="display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; padding: 1rem 0;">
"""
phases = [
    ("1a", "Data Loading\n& Cleaning", "#6C9BF2"),
    ("1b", "Exploratory\nData Analysis", "#6C9BF2"),
    ("2", "Construct\nValidation", "#B088F9"),
    ("3", "Inferential\nModelling", "#FFB84D"),
    ("4", "Predictive AI\nModelling", "#00D4AA"),
    ("5", "MCDM Scenario\nAnalysis", "#FF6B6B"),
    ("6", "Validation &\nSelf-Challenge", "#8892A4"),
]

for num, label, color in phases:
    flow_html += f"""
    <div style="
        background: rgba(26,31,46,0.7);
        border: 1px solid {color}40;
        border-radius: 10px;
        padding: 16px 20px;
        min-width: 130px;
        text-align: center;
        flex: 1;
    ">
        <div style="font-size: 1.5rem; font-weight: 700; color: {color};">Phase {num}</div>
        <div style="font-size: 0.8rem; color: #8892A4; white-space: pre-line; margin-top: 6px;">{label}</div>
    </div>
    """

flow_html += "</div>"
st.markdown(flow_html, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Dataset Description ──────────────────────────────────────────────
st.markdown("### Dataset Summary")

ds_cols = st.columns(4)
ds_items = [
    ("199", "Respondents"),
    ("3", "Countries"),
    ("6", "BIM Phases"),
    ("3", "Sustainability Targets"),
]
for col, (val, label) in zip(ds_cols, ds_items):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{val}</div>'
        f'<div class="metric-label">{label}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("")
st.markdown(
    """
| Variable Group | Items | Scale |
|:---|:---|:---|
| BIM-use per lifecycle phase | 6 items | Likert 1–5 |
| Sustainability outcomes | 3 items | Likert 1–5 |
| Control variables | Country, company size, project size, legal form, etc. | Categorical |
| Participant metadata | Status, foreign participation, countries operated in | Mixed |
"""
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Model Architecture Summary ───────────────────────────────────────
st.markdown("### Model Architecture")

st.markdown(
    """
| Component | Detail |
|:---|:---|
| **Feature Engineering** | BIM maturity index, lifecycle coverage, EOL focus, interaction terms, one-hot country/status |
| **ML Models** | Random Forest, XGBoost (GPU), LightGBM (GPU), SVM (RBF), Ordinal Ridge, LAD |
| **Hyperparameter Tuning** | Inner GridSearchCV (5-fold) per model per target |
| **Validation** | 5×5 Repeated Stratified K-Fold CV (25 folds) |
| **Interpretability** | SHAP TreeExplainer — mean |SHAP| per feature |
| **MCDM** | TOPSIS with vector normalisation; 4 weighting schemes |
"""
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Limitations ───────────────────────────────────────────────────────
st.markdown("### Limitations")

st.markdown(
    '<div class="callout callout-warn">'
    "<strong>Key limitations to consider:</strong>"
    "<ul>"
    "<li>Sustainability targets are <strong>perceived impacts</strong> (survey-based), not "
    "measured LCA/GWP values.</li>"
    "<li>The dataset contains only <strong>9 unique respondent profiles</strong> across 199 rows, "
    "which significantly limits effective sample diversity and model generalisability.</li>"
    "<li>Perfect cross-validation scores (F1 = 1.0) indicate potential overfitting to the "
    "limited profile set — models may not generalise to unseen respondent types.</li>"
    "<li>Country subgroups are unbalanced: Croatia (99), Slovakia (60), Slovenia (40).</li>"
    "<li>Ordinal targets (1–5 Likert) have limited granularity for regression approaches.</li>"
    "<li>The Circularity sub-construct (2 items) has low internal consistency (α = 0.37).</li>"
    "</ul>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Citation ──────────────────────────────────────────────────────────
st.markdown("### Citation")

st.code(
    """@article{bim_lca_circularity_2025,
  title   = {AI-Based Predictive Integrated Model for BIM--LCA--Circularity 
             Assessment in Construction Projects},
  author  = {[Author Names]},
  journal = {[Target Journal]},
  year    = {2025},
  note    = {Manuscript in preparation}
}""",
    language="bibtex",
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Technology Stack ──────────────────────────────────────────────────
st.markdown("### Technology Stack")

tech_cols = st.columns(4)
tech_items = [
    ("🐍 Python", "Core language"),
    ("📊 Streamlit", "Web framework"),
    ("🤖 scikit-learn", "ML models"),
    ("📈 Plotly", "Visualisations"),
]
for col, (tech, desc) in zip(tech_cols, tech_items):
    col.markdown(
        f'<div class="metric-card" style="padding: 1rem;">'
        f'<div style="font-size: 1.1rem; font-weight: 600; color: #E8ECF1;">{tech}</div>'
        f'<div style="font-size: 0.8rem; color: #8892A4; margin-top: 4px;">{desc}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Footer ────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    "<p>Built with <a href='https://streamlit.io' target='_blank'>Streamlit</a> · "
    "Models: scikit-learn, XGBoost, LightGBM · "
    "Interpretability: SHAP · "
    "MCDM: TOPSIS</p>"
    "</div>",
    unsafe_allow_html=True,
)
