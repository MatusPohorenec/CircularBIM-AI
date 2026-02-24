"""🤖 Predictor — Interactive BIM sustainability prediction tool."""
import streamlit as st
import numpy as np

from src.st_config import (
    APP_TITLE,
    APP_ICON,
    CSS_PATH,
    BIM_PHASE_LABELS,
    TARGET_LABELS,
    TARGET_ICONS,
    COUNTRY_FLAGS,
    PROJECT_SIZE_OPTIONS,
    COMPANY_SIZE_OPTIONS,
    C_ACCENT,
    C_DANGER,
    C_WARN,
)
from src.charts import create_gauge, create_proba_bar
from src.model import build_feature_vector, predict_all, load_model

st.set_page_config(page_title=f"{APP_TITLE} — Predictor", page_icon="🤖", layout="wide")

if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"### {APP_ICON} {APP_TITLE}")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("## 🤖 Sustainability Predictor")
st.markdown(
    '<p style="color: #8892A4;">Configure a project\'s BIM adoption profile and '
    "predict its sustainability and circularity outcomes.</p>",
    unsafe_allow_html=True,
)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Verify models available ──────────────────────────────────────────
try:
    _ = load_model("sust_recycling_rate")
    models_available = True
except Exception:
    models_available = False
    st.warning(
        "⚠️ Pre-trained models not found. Run `python scripts/export_models.py` "
        "to generate model files before using the predictor."
    )


# ── Input Form (left) | Results (right) ──────────────────────────────
col_input, col_result = st.columns([2, 3])

with col_input:
    st.markdown("### Project Configuration")

    # Country
    country_options = list(COUNTRY_FLAGS.keys())
    country_display = [f"{COUNTRY_FLAGS[c]} {c}" for c in country_options]
    country_idx = st.selectbox("Country", range(len(country_options)),
                               format_func=lambda i: country_display[i])
    country = country_options[country_idx]

    # Participant status
    participant_status = st.selectbox(
        "Participant Status",
        ["Contractor (main)", "Designer", "Investor"],
    )

    # Company size
    company_size = st.select_slider(
        "Company Size",
        options=COMPANY_SIZE_OPTIONS,
        value=COMPANY_SIZE_OPTIONS[1],
    )

    # Project size
    project_size = st.select_slider(
        "Project Size",
        options=PROJECT_SIZE_OPTIONS,
        value=PROJECT_SIZE_OPTIONS[1],
    )

    # Foreign participation
    foreign_participation = st.toggle("Foreign Participation", value=True)

    # Countries operated in
    countries_operated_in = st.slider("Countries Operated In", 1, 6, 2)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # BIM-use per phase — lifecycle timeline
    st.markdown("### BIM Adoption per Lifecycle Phase")
    st.caption("Rate BIM usage (1 = None → 5 = Full) for each building lifecycle phase.")

    bim_values: dict[str, int] = {}
    phase_icons = ["📋", "⚡", "🏗️", "📐", "🔨", "♻️"]

    for (col_name, label), icon in zip(BIM_PHASE_LABELS.items(), phase_icons):
        bim_values[col_name] = st.slider(
            f"{icon} {label}",
            min_value=1,
            max_value=5,
            value=3,
            key=f"bim_{col_name}",
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Predict button
    predict_clicked = st.button("🔮 Predict Sustainability Impact", use_container_width=True, type="primary")


with col_result:
    st.markdown("### Prediction Results")

    if not models_available:
        st.info("Load the pre-trained models to see predictions.")

    elif predict_clicked or st.session_state.get("_last_prediction"):
        # Build feature vector  
        try:
            feature_df = build_feature_vector(
                bim_values=bim_values,
                country=country,
                participant_status=participant_status,
                project_size=project_size,
                company_size=company_size,
                foreign_participation=foreign_participation,
                countries_operated_in=countries_operated_in,
            )

            results = predict_all(feature_df)
            st.session_state["_last_prediction"] = results
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            results = st.session_state.get("_last_prediction")

        if results:
            # Display gauges
            gauge_cols = st.columns(3)
            for col_g, (target_key, target_label) in zip(gauge_cols, TARGET_LABELS.items()):
                with col_g:
                    r = results[target_key]
                    pred = r["prediction"]
                    icon = TARGET_ICONS.get(target_key, "")

                    st.markdown(f"**{icon} {target_label}**")
                    fig_gauge = create_gauge(pred, target_label)
                    st.plotly_chart(fig_gauge, width="stretch", key=f"gauge_{target_key}")

                    # Probability bar
                    proba = r["probabilities"]
                    if proba:
                        fig_bar = create_proba_bar(proba)
                        st.plotly_chart(fig_bar, width="stretch", key=f"proba_{target_key}")

                    # Expected value
                    st.caption(f"Expected value: **{r['expected']:.2f}** / 5.00")

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # Interpretation
            st.markdown("### 📝 Interpretation")

            bim_vals_list = list(bim_values.values())
            mean_bim = np.mean(bim_vals_list)
            max_phase = max(bim_values, key=bim_values.get)
            min_phase = min(bim_values, key=bim_values.get)
            max_label = BIM_PHASE_LABELS[max_phase]
            min_label = BIM_PHASE_LABELS[min_phase]

            recycling_pred = results["sust_recycling_rate"]["prediction"]
            waste_pred = results["sust_waste_reduction"]["prediction"]
            co2_pred = results["sust_co2_reduction"]["prediction"]

            level_word = {1: "very low", 2: "low", 3: "moderate", 4: "high", 5: "very high"}

            interpretation = (
                f"Based on the configured BIM adoption profile (mean BIM use: {mean_bim:.1f}/5), "
                f"this project shows **{level_word.get(co2_pred, 'moderate')}** potential for CO₂ reduction "
                f"(level {co2_pred}/5), **{level_word.get(waste_pred, 'moderate')}** waste reduction "
                f"(level {waste_pred}/5), and **{level_word.get(recycling_pred, 'moderate')}** recycling "
                f"rate improvement (level {recycling_pred}/5).\n\n"
                f"**Key driver:** highest BIM use in *{max_label}* phase "
                f"(score {bim_values[max_phase]}/5). "
            )

            if bim_values.get("bim_demolition", 1) <= 2 and bim_values.get("bim_recycling", 1) <= 2:
                interpretation += (
                    "Low BIM adoption in end-of-life phases (Demolition, Recycling) limits "
                    "circularity outcomes. Consider increasing BIM use in these stages."
                )
            elif mean_bim >= 4:
                interpretation += (
                    "High BIM adoption across all phases supports strong sustainability outcomes."
                )

            st.markdown(
                f'<div class="callout">{interpretation}</div>',
                unsafe_allow_html=True,
            )

    else:
        st.markdown(
            '<div class="callout">'
            "Configure your project parameters on the left and click "
            '<strong>"Predict"</strong> to see sustainability predictions.'
            "</div>",
            unsafe_allow_html=True,
        )


# ── Footer ────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    "<p>Predictions are based on Random Forest models trained on survey data from 199 respondents "
    "across Slovakia, Croatia, and Slovenia. Results reflect perceived BIM impacts.</p>"
    "</div>",
    unsafe_allow_html=True,
)
