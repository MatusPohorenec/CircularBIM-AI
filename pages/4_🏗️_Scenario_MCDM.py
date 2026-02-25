"""🏗️ Scenario & MCDM — Compare BIM adoption strategies using TOPSIS MCDM."""
import json
import streamlit as st
import numpy as np
import pandas as pd

from src.st_config import (
    APP_TITLE,
    APP_ICON,
    ASSETS_DIR,
    CSS_PATH,
    BIM_PHASE_LABELS,
    TARGET_LABELS,
    C_ACCENT,
    C_DANGER,
    C_WARN,
    C_INFO,
    C_MUTED,
    COLORWAY,
)
from src.charts import create_radar, create_heatmap, create_horizontal_bar

st.set_page_config(page_title=f"{APP_TITLE} — Scenario MCDM", page_icon="🏗️", layout="wide")

st.markdown('<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet">', unsafe_allow_html=True)
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_json(name: str) -> dict:
    path = ASSETS_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


mcdm_data = load_json("mcdm_results.json")

with st.sidebar:
    st.markdown(f"### {APP_ICON} {APP_TITLE}")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("## 🏗️ Scenario Comparison & MCDM")
st.markdown(
    '<p style="color: #8892A4;">Compare BIM adoption strategies using TOPSIS multi-criteria '
    "decision making to identify optimal configurations.</p>",
    unsafe_allow_html=True,
)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Pre-defined scenarios ─────────────────────────────────────────────
SCENARIOS = {
    "S1: Baseline (Low BIM)": [1, 1, 2, 1, 1, 1],
    "S2: Design-Focused BIM": [4, 5, 3, 2, 1, 1],
    "S3: Construction-Focused BIM": [2, 3, 5, 4, 1, 1],
    "S4: End-of-Life Focused BIM": [2, 2, 3, 2, 5, 5],
    "S5: Full Lifecycle BIM": [4, 5, 5, 4, 4, 4],
    "S6: Maximum BIM Adoption": [5, 5, 5, 5, 5, 5],
}
BIM_PHASES = list(BIM_PHASE_LABELS.values())
BIM_COLS = list(BIM_PHASE_LABELS.keys())


# ── TOPSIS Implementation ────────────────────────────────────────────
def topsis(dm: np.ndarray, weights: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    """TOPSIS ranking — returns closeness coefficients."""
    norms = np.sqrt((dm ** 2).sum(axis=0))
    norms[norms == 0] = 1
    norm_matrix = dm / norms
    weighted = norm_matrix * weights
    ideal = np.where(benefit, weighted.max(axis=0), weighted.min(axis=0))
    anti_ideal = np.where(benefit, weighted.min(axis=0), weighted.max(axis=0))
    d_ideal = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_anti = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))
    return d_anti / (d_ideal + d_anti + 1e-10)


# ── Scenario Builder ─────────────────────────────────────────────────
st.markdown("### Scenario Definitions")

use_custom = st.checkbox("Add a Custom Scenario")

custom_scenario: dict[str, int] | None = None
if use_custom:
    st.markdown("**Custom Scenario — BIM Levels:**")
    custom_cols = st.columns(6)
    custom_vals = []
    for col_ui, (phase_key, phase_label) in zip(custom_cols, BIM_PHASE_LABELS.items()):
        with col_ui:
            v = st.number_input(phase_label, min_value=1, max_value=5, value=3, key=f"custom_{phase_key}")
            custom_vals.append(v)
    custom_scenario = dict(zip(BIM_COLS, custom_vals))

# Build scenario display table
all_scenarios = dict(SCENARIOS)
if custom_scenario:
    all_scenarios["S7: Custom"] = list(custom_scenario.values())

# Radar chart of all scenarios  
st.markdown("### BIM Adoption Profiles")
radar_data = {name: vals for name, vals in all_scenarios.items()}
fig_radar = create_radar(radar_data, BIM_PHASES)
st.plotly_chart(fig_radar, width="stretch", key="mcdm_radar")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── MCDM Configuration ───────────────────────────────────────────────
st.markdown("### MCDM Weight Configuration")

criteria_labels = ["♻️ Recycling Rate", "🗑️ Waste Reduction", "🌍 CO₂ Reduction"]

weight_cols = st.columns(3)
raw_weights = []
for col_w, label in zip(weight_cols, criteria_labels):
    with col_w:
        w = st.slider(label, 0.0, 1.0, 1 / 3, 0.05, key=f"w_{label}")
        raw_weights.append(w)

# Normalise
total = sum(raw_weights)
if total > 0:
    weights = np.array([w / total for w in raw_weights])
else:
    weights = np.array([1 / 3] * 3)

st.caption(f"Normalised weights: {weights[0]:.2f} / {weights[1]:.2f} / {weights[2]:.2f} (sum = 1.00)")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Run MCDM ─────────────────────────────────────────────────────────
st.markdown("### TOPSIS Ranking")

# Get predictions from pre-computed data
predictions = mcdm_data.get("predictions", [])
prediction_map = {p["scenario"]: p for p in predictions}

# Build decision matrix from pre-computed or scenario defaults
scenario_names = list(all_scenarios.keys())
dm_rows = []
for s_name in scenario_names:
    pred = prediction_map.get(s_name)
    if pred:
        dm_rows.append([
            pred.get("sust_recycling_rate_expected", pred.get("sust_recycling_rate", 1)),
            pred.get("sust_waste_reduction_expected", pred.get("sust_waste_reduction", 1)),
            pred.get("sust_co2_reduction_expected", pred.get("sust_co2_reduction", 1)),
        ])
    else:
        # For custom scenario, use simple BIM average as proxy
        vals = all_scenarios[s_name]
        mean_bim = np.mean(vals)
        dm_rows.append([mean_bim, mean_bim, mean_bim])

dm = np.array(dm_rows)
benefit = np.array([True, True, True])
cc = topsis(dm, weights, benefit)
ranks = np.argsort(-cc) + 1

# Build ranking table
rank_df = pd.DataFrame({
    "Rank": ranks,
    "Scenario": scenario_names,
    "TOPSIS Score": cc,
    "Recycling (exp)": dm[:, 0],
    "Waste (exp)": dm[:, 1],
    "CO₂ (exp)": dm[:, 2],
}).sort_values("Rank")

# Display as styled HTML table
header_html = "".join(f"<th>{c}</th>" for c in rank_df.columns)
rows_html = ""
for _, row in rank_df.iterrows():
    row_class = ""
    cells = ""
    for c in rank_df.columns:
        val = row[c]
        cell_class = ""
        if c == "Rank" and val == 1:
            cell_class = ' class="highlight-cell"'
        if isinstance(val, float):
            val_str = f"{val:.3f}"
        else:
            val_str = str(val)
        cells += f"<td{cell_class}>{val_str}</td>"
    rows_html += f"<tr>{cells}</tr>"

st.markdown(
    f'<table class="styled-table"><thead><tr>{header_html}</tr></thead>'
    f"<tbody>{rows_html}</tbody></table>",
    unsafe_allow_html=True,
)

# Highlight winner
winner_idx = np.argmax(cc)
winner_name = scenario_names[winner_idx]
st.success(f"🏆 **Top-ranked scenario:** {winner_name} (TOPSIS score: {cc[winner_idx]:.4f})")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Top 3 Radar Overlay ──────────────────────────────────────────────
st.markdown("### Top 3 Scenarios — BIM Profile Comparison")

top3_indices = np.argsort(-cc)[:3]
top3_data = {scenario_names[i]: all_scenarios[scenario_names[i]] for i in top3_indices}
fig_top3 = create_radar(top3_data, BIM_PHASES)
st.plotly_chart(fig_top3, width="stretch", key="top3_radar")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Sensitivity Analysis ─────────────────────────────────────────────
st.markdown("### Sensitivity Analysis")
st.caption("How stable are rankings when criterion weights vary by ±20%?")

sensitivity_data = mcdm_data.get("sensitivity", [])

if sensitivity_data:
    sens_df = pd.DataFrame(sensitivity_data)
    criteria_names = ["Recycling Rate", "Waste Reduction", "CO₂ Reduction"]

    sens_tabs = st.tabs(criteria_names)
    for tab, crit_idx, crit_name in zip(sens_tabs, range(3), criteria_names):
        with tab:
            crit_data = sens_df[sens_df["criterion_varied"] == crit_idx]
            if not crit_data.empty:
                # Pivot to heatmap
                pivot = crit_data.pivot_table(
                    values="rank", index="scenario", columns="delta"
                )
                # Sort columns
                pivot = pivot[sorted(pivot.columns)]
                col_labels = [f"{d:+.0%}" for d in sorted(pivot.columns)]

                fig_sens = create_heatmap(
                    z=pivot.values.tolist(),
                    x_labels=col_labels,
                    y_labels=pivot.index.tolist(),
                    colorscale=[[0, C_ACCENT], [0.5, C_WARN], [1, C_DANGER]],
                    zmin=1,
                    zmax=6,
                    fmt=".0f",
                )
                fig_sens.update_layout(height=max(250, len(pivot) * 40 + 80))
                st.plotly_chart(fig_sens, width="stretch", key=f"sens_{crit_idx}")
            else:
                st.caption("No sensitivity data for this criterion.")

    with st.expander("📖 Interpretation"):
        st.markdown(
            "The sensitivity heatmap shows how scenario rankings change when each criterion "
            "weight is varied by ±20%. Rankings that remain stable (same rank across all "
            "perturbations) indicate robust MCDM decisions. The top scenarios (S5, S6) "
            "consistently rank highest regardless of weight variations, confirming the "
            "robustness of the Full/Maximum BIM adoption strategies."
        )
else:
    st.info("Sensitivity data not available. Run the full pipeline first.")


# ── Footer ────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    "<p>MCDM analysis uses TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution). "
    "Decision matrix values are expected sustainability scores from RF model predictions.</p>"
    "</div>",
    unsafe_allow_html=True,
)
