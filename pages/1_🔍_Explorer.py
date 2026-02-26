"""🔍 Data Explorer — Interactive EDA dashboard (aggregate data only)."""
import json
import streamlit as st

from src.st_config import (
    APP_TITLE,
    APP_ICON,
    ASSETS_DIR,
    CSS_PATH,
    BIM_PHASE_LABELS,
    TARGET_LABELS,
    COUNTRY_FLAGS,
    C_ACCENT,
    C_DANGER,
    C_WARN,
    C_INFO,
    C_MUTED,
)
from src.charts import create_heatmap, create_radar, create_horizontal_bar, create_bar_chart

st.set_page_config(page_title=f"{APP_TITLE} — Explorer", page_icon="🔍", layout="wide")

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


eda = load_json("eda_stats.json")
distributions = load_json("distributions.json")
correlations = load_json("correlations.json")
bim_heatmap = load_json("bim_heatmap.json")

if not eda:
    st.warning("Pre-computed EDA assets not found. Run `scripts/export_models.py` first.")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### {APP_ICON} {APP_TITLE}")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

countries_available = list(eda.get("countries", {}).keys())
all_countries = ["All"] + countries_available

selected_country = st.sidebar.selectbox(
    "🌍 Country Filter",
    all_countries,
    index=0,
)


# ── Page header ───────────────────────────────────────────────────────
st.markdown("## 🔍 Data Explorer")
st.markdown(
    '<p style="color: #8892A4;">Interactive exploration of aggregate BIM adoption '
    'and sustainability data across Slovakia, Croatia, and Slovenia.</p>',
    unsafe_allow_html=True,
)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── KPI overview ──────────────────────────────────────────────────────
countries_dict = eda.get("countries", {})
if selected_country == "All":
    n_display = eda.get("n_rows", 0)
    countries_label = f"{len(countries_dict)} countries"
else:
    n_display = countries_dict.get(selected_country, 0)
    countries_label = f"{COUNTRY_FLAGS.get(selected_country, '')} {selected_country}"

cols = st.columns(4)
kpi = [
    (str(n_display), "Respondents", countries_label),
    (str(eda.get("n_cols", 0)), "Variables", "18 survey items"),
    (str(len(countries_dict)), "Countries", "SK, HR, SI"),
    ("6", "BIM Phases", "Full lifecycle"),
]
for col, (val, label, delta) in zip(cols, kpi):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{val}</div>'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-delta-up">{delta}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── BIM Adoption Heatmap ─────────────────────────────────────────────
st.markdown("### BIM Adoption by Country & Lifecycle Phase")

if bim_heatmap:
    phase_labels = bim_heatmap.get("phase_labels", bim_heatmap.get("phases", []))
    hm_countries = bim_heatmap.get("countries", [])
    hm_values = bim_heatmap.get("values", [])

    fig_hm = create_heatmap(
        z=hm_values,
        x_labels=phase_labels,
        y_labels=[f"{COUNTRY_FLAGS.get(c, '')} {c}" for c in hm_countries],
        colorscale=[[0, "#0E1117"], [0.5, "#1A5C4B"], [1, C_ACCENT]],
        zmin=1,
        zmax=5,
    )
    st.plotly_chart(fig_hm, width="stretch", key="bim_heatmap")
else:
    st.info("BIM heatmap data not available.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Sustainability Target Distributions ───────────────────────────────
st.markdown("### Sustainability Outcome Distributions")

target_keys = list(TARGET_LABELS.keys())
dist_cols = st.columns(len(target_keys))

for col, target in zip(dist_cols, target_keys):
    with col:
        st.markdown(f"**{TARGET_LABELS[target]}**")
        target_dist = distributions.get(target, {})

        if selected_country == "All":
            dist_data = target_dist.get("All", {})
        else:
            dist_data = target_dist.get(selected_country, {})

        if dist_data:
            classes = sorted(dist_data.keys(), key=lambda x: int(x))
            counts = [dist_data[c] for c in classes]
            class_labels = [f"Level {c}" for c in classes]
            fig = create_bar_chart(
                x=class_labels,
                y=counts,
                color=C_ACCENT,
                height=280,
            )
            st.plotly_chart(fig, width="stretch", key=f"dist_{target}")
        else:
            st.caption("No data for this filter.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Radar / Spider Chart ─────────────────────────────────────────────
st.markdown("### BIM-Use Profile — Radar Chart")

if bim_heatmap:
    phase_labels = bim_heatmap.get("phase_labels", [])
    hm_countries = bim_heatmap.get("countries", [])
    hm_values = bim_heatmap.get("values", [])

    if selected_country == "All":
        radar_data = {
            f"{COUNTRY_FLAGS.get(c, '')} {c}": vals
            for c, vals in zip(hm_countries, hm_values)
        }
    else:
        idx = hm_countries.index(selected_country) if selected_country in hm_countries else None
        if idx is not None:
            radar_data = {f"{COUNTRY_FLAGS.get(selected_country, '')} {selected_country}": hm_values[idx]}
        else:
            radar_data = {}

    if radar_data:
        fig_radar = create_radar(radar_data, phase_labels)
        st.plotly_chart(fig_radar, width="stretch", key="bim_radar")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Correlation Matrix ────────────────────────────────────────────────
st.markdown("### Spearman Correlation Matrix")

if correlations:
    corr_cols = correlations.get("columns", [])
    corr_vals = correlations.get("values", [])

    short_labels = [
        "BIM:Feasibility",
        "BIM:Energy",
        "BIM:Construction",
        "BIM:Space",
        "BIM:Demolition",
        "BIM:Recycling",
        "Sust:Recycling",
        "Sust:Waste",
        "Sust:CO₂",
    ]
    display_labels = short_labels[: len(corr_cols)]

    fig_corr = create_heatmap(
        z=corr_vals,
        x_labels=display_labels,
        y_labels=display_labels,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    st.plotly_chart(fig_corr, width="stretch", key="corr_heatmap")

    with st.expander("📖 Interpretation"):
        st.markdown(
            "Spearman rank correlations capture monotonic relationships between ordinal BIM-use "
            "and sustainability indicators. High correlations between `BIM:Energy` → `Sust:Waste` "
            "(ρ ≈ 0.91) and `BIM:Recycling` → `Sust:CO₂` (ρ ≈ 0.72) suggest that early design-phase "
            "BIM adoption and end-of-life phase practices are associated with better sustainability outcomes."
        )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Categorical Breakdowns ───────────────────────────────────────────
st.markdown("### Sample Composition")

cat_data = eda.get("categorical", {})
show_cats = ["participant_status", "project_size", "company_size", "foreign_participation"]

cat_cols = st.columns(2)
for i, cat_key in enumerate(show_cats):
    if cat_key in cat_data:
        with cat_cols[i % 2]:
            st.markdown(f"**{cat_key.replace('_', ' ').title()}**")
            data = cat_data[cat_key]
            labels = list(data.keys())
            values = list(data.values())

            # Show percentages for all categories except project_size
            if cat_key != "project_size":
                total = sum(values)
                pct_values = [round(v / total * 100, 1) if total else 0 for v in values]
                fig = create_bar_chart(x=labels, y=pct_values, color=C_INFO, height=250)
                fig.update_layout(yaxis_title="%", yaxis_ticksuffix=" %")
            else:
                fig = create_bar_chart(x=labels, y=values, color=C_INFO, height=250)

            st.plotly_chart(fig, width="stretch", key=f"cat_{cat_key}")


# ── Footer ────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    "<p>All visualisations show aggregate statistics. No individual respondent data is exposed.</p>"
    "</div>",
    unsafe_allow_html=True,
)
