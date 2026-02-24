"""📊 Model Insights — Performance metrics, SHAP feature importance, and interpretability."""
import json
import streamlit as st
import pandas as pd
import numpy as np

from src.st_config import (
    APP_TITLE,
    APP_ICON,
    ASSETS_DIR,
    CSS_PATH,
    TARGET_LABELS,
    TARGET_ICONS,
    C_ACCENT,
    C_DANGER,
    C_WARN,
    C_INFO,
    C_MUTED,
    C_TEXT,
    COLORWAY,
)
from src.charts import create_heatmap, create_horizontal_bar

st.set_page_config(page_title=f"{APP_TITLE} — Model Insights", page_icon="📊", layout="wide")

if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_json(name: str) -> dict | list:
    path = ASSETS_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


model_results = load_json("model_results.json")
shap_data = load_json("shap_data.json")
reliability = load_json("reliability.json")
sem_results = load_json("sem_results.json")

with st.sidebar:
    st.markdown(f"### {APP_ICON} {APP_TITLE}")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("## 📊 Model Insights")
st.markdown(
    '<p style="color: #8892A4;">Transparency layer: model performance, feature importance '
    "(SHAP), and construct reliability.</p>",
    unsafe_allow_html=True,
)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Model Comparison Table ────────────────────────────────────────────
st.markdown("### Model Comparison")

if model_results:
    df_models = pd.DataFrame(model_results)

    # Only show key metrics
    display_cols = ["target", "model", "macro_f1_mean", "macro_f1_std",
                    "cohen_kappa_qw_mean", "mae_mean", "auc_roc_ovr_mean"]
    avail_cols = [c for c in display_cols if c in df_models.columns]
    df_display = df_models[avail_cols].copy()

    # Rename columns for display
    rename_map = {
        "target": "Target",
        "model": "Model",
        "macro_f1_mean": "Macro-F1",
        "macro_f1_std": "F1 Std",
        "cohen_kappa_qw_mean": "Weighted κ",
        "mae_mean": "MAE",
        "auc_roc_ovr_mean": "AUC-ROC",
    }
    df_display = df_display.rename(columns=rename_map)

    # Clean target names for display
    target_display = {
        "sust_recycling_rate": "♻️ Recycling",
        "sust_waste_reduction": "🗑️ Waste",
        "sust_co2_reduction": "🌍 CO₂",
    }
    if "Target" in df_display.columns:
        df_display["Target"] = df_display["Target"].map(target_display).fillna(df_display["Target"])

    # Filter by target
    target_filter = st.selectbox(
        "Filter by target",
        ["All"] + list(target_display.values()),
    )

    if target_filter != "All":
        df_display = df_display[df_display["Target"] == target_filter]

    # Display styled table
    st.markdown(
        '<div style="overflow-x: auto;">',
        unsafe_allow_html=True,
    )

    # Build HTML table  
    header_html = "".join(f"<th>{c}</th>" for c in df_display.columns)
    rows_html = ""
    for _, row in df_display.iterrows():
        cells = ""
        for col_name in df_display.columns:
            val = row[col_name]
            cell_class = ""
            if col_name == "Macro-F1" and isinstance(val, (int, float)) and not pd.isna(val):
                if val >= 0.95:
                    cell_class = ' class="highlight-cell"'
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            elif isinstance(val, float) and not pd.isna(val):
                val_str = f"{val:.4f}"
            elif pd.isna(val) if isinstance(val, float) else val is None:
                val_str = "—"
            else:
                val_str = str(val)
            cells += f"<td{cell_class}>{val_str}</td>"
        rows_html += f"<tr>{cells}</tr>"

    table_html = f"""
    <table class="styled-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📖 Interpretation"):
        st.markdown(
            "All ML models (Random Forest, XGBoost, LightGBM, SVM) achieve perfect "
            "cross-validated macro-F1 = 1.00 across all three targets. This reflects the "
            "high separability of the ordinal classes given the dataset's structure "
            "(9 unique respondent profiles across 199 observations). While impressive, "
            "this should be interpreted cautiously — see the Validation section for "
            "overfitting caveats."
        )
else:
    st.info("Model comparison results not available.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Construct Reliability ─────────────────────────────────────────────
st.markdown("### Construct Reliability")

if reliability:
    rel_df = pd.DataFrame(reliability)

    header_html = "".join(f"<th>{c}</th>" for c in rel_df.columns)
    rows_html = ""
    for _, row in rel_df.iterrows():
        cells = ""
        for c in rel_df.columns:
            val = row[c]
            cell_class = ""
            if c == "Acceptable (α≥0.60)" and str(val) == "NO":
                cell_class = ' style="color: #FF6B6B; font-weight: 600;"'
            elif c == "Acceptable (α≥0.60)" and str(val) == "Yes":
                cell_class = ' style="color: #00D4AA; font-weight: 600;"'
            cells += f"<td{cell_class}>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"

    st.markdown(
        f'<table class="styled-table"><thead><tr>{header_html}</tr></thead>'
        f"<tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True,
    )

    with st.expander("📖 Interpretation"):
        st.markdown(
            "**BIM-use maturity** (α = 0.75) and **Sustainability impact** (α = 0.86) show "
            "acceptable internal consistency. The **Circularity sub-construct** (α = 0.37) has "
            "low reliability, which is expected for a 2-item scale. McDonald's omega (ω) provides "
            "a more robust estimate, showing ω = 0.83 and ω = 0.92 for the main constructs."
        )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── SHAP Feature Importance ──────────────────────────────────────────
st.markdown("### SHAP Feature Importance")

if shap_data:
    target_tabs = st.tabs([
        f"{TARGET_ICONS.get(t, '')} {TARGET_LABELS.get(t, t)}"
        for t in TARGET_LABELS
    ])

    for tab, (target_key, target_label) in zip(target_tabs, TARGET_LABELS.items()):
        with tab:
            features = shap_data.get(target_key, [])
            if features:
                top_n = min(15, len(features))
                top_features = features[:top_n]
                labels = [f["feature"].replace("_", " ").title() for f in reversed(top_features)]
                values = [f["importance"] for f in reversed(top_features)]

                fig = create_horizontal_bar(
                    labels=labels,
                    values=values,
                    color=C_ACCENT,
                    height=max(300, top_n * 28),
                )
                fig.update_layout(yaxis=dict(autorange=True))
                st.plotly_chart(fig, width="stretch", key=f"shap_{target_key}")

                with st.expander("📖 Key drivers"):
                    top3 = features[:3]
                    drivers = ", ".join(
                        f"**{f['feature'].replace('_', ' ').title()}** "
                        f"(|SHAP| = {f['importance']:.4f})"
                        for f in top3
                    )
                    st.markdown(
                        f"The top 3 features driving {target_label} predictions are: {drivers}. "
                        "Higher mean |SHAP| indicates the feature has greater influence on "
                        "the model's predictions across all observations."
                    )
            else:
                st.caption("SHAP data not available for this target.")

else:
    st.info("SHAP data not available. Run `scripts/export_models.py` first.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── SEM Path Results ──────────────────────────────────────────────────
st.markdown("### PLS-SEM Path Coefficients")

if sem_results and "estimates" in sem_results:
    estimates = sem_results["estimates"]
    structural = [e for e in estimates if e.get("op") == "~" and e.get("lval") in ["Sust_impact"]]

    if structural:
        st.markdown("**Structural paths (BIM Maturity → Sustainability Impact):**")
        for path in structural:
            rval = path["rval"]
            estimate = path.get("Estimate", "—")
            p_value = path.get("p-value", "—")

            try:
                p_float = float(p_value)
                sig = "***" if p_float < 0.001 else "**" if p_float < 0.01 else "*" if p_float < 0.05 else "ns"
                p_display = f"{p_float:.4f}"
            except (ValueError, TypeError):
                sig = ""
                p_display = str(p_value)

            rval_display = rval.replace("_", " ").title()
            color = C_ACCENT if sig and sig != "ns" else C_MUTED

            st.markdown(
                f'<div class="callout" style="border-left-color: {color};">'
                f"<strong>{rval_display}</strong> → Sustainability Impact: "
                f"β = {estimate}, p = {p_display} {sig}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Fit statistics
    if "fit" in sem_results and sem_results["fit"]:
        fit = sem_results["fit"][0]
        with st.expander("📊 SEM Fit Statistics"):
            fit_items = {
                "CFI": fit.get("CFI", "—"),
                "RMSEA": fit.get("RMSEA", "—"),
                "χ²": fit.get("chi2", "—"),
                "DoF": fit.get("DoF", "—"),
                "AIC": fit.get("AIC", "—"),
                "BIC": fit.get("BIC", "—"),
            }
            cols = st.columns(len(fit_items))
            for col, (label, val) in zip(cols, fit_items.items()):
                if isinstance(val, float):
                    val_str = f"{val:.4f}"
                else:
                    val_str = str(val)
                col.metric(label, val_str)

            st.caption(
                "Note: The low CFI (0.38) and high RMSEA (0.58) indicate poor model fit, "
                "which is expected given the small sample size and ordinal data characteristics. "
                "PLS-SEM results should be interpreted as exploratory."
            )


# ── Footer ────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    "<p>Model transparency: all metrics computed via 5×5 repeated stratified k-fold cross-validation. "
    "SHAP values computed using TreeExplainer on the Random Forest model.</p>"
    "</div>",
    unsafe_allow_html=True,
)
