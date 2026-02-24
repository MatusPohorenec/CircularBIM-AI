"""Streamlit app configuration — centralised constants, paths, chart templates."""
from pathlib import Path

APP_TITLE = "BIM–LCA–Circularity AI"
APP_ICON = "🏗️"
APP_SUBTITLE = "AI-Based Predictive Model for Multicriteria Sustainability Assessment"

# Paths (relative to repo root for Streamlit Cloud compatibility)
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
CSS_PATH = ASSETS_DIR / "style.css"

# Colour palette
C_ACCENT = "#00D4AA"
C_DANGER = "#FF6B6B"
C_WARN = "#FFB84D"
C_INFO = "#6C9BF2"
C_PURPLE = "#B088F9"
C_SURFACE1 = "#1A1F2E"
C_SURFACE2 = "#232A3B"
C_TEXT = "#E8ECF1"
C_MUTED = "#8892A4"
C_BG = "#0E1117"

COLORWAY = [C_ACCENT, C_DANGER, C_WARN, C_INFO, C_PURPLE]

# Plotly template
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color=C_TEXT, size=13),
    colorway=COLORWAY,
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    margin=dict(l=20, r=20, t=40, b=20),
)

# Feature labels for display
BIM_PHASE_LABELS = {
    "bim_feasibility_study": "Feasibility Study",
    "bim_energy_sustainability": "Energy Sustainability",
    "bim_construction_mgmt": "Construction Mgmt",
    "bim_space_tracking": "Space Tracking",
    "bim_demolition": "Demolition",
    "bim_recycling": "Recycling (Circularity)",
}

TARGET_LABELS = {
    "sust_recycling_rate": "Recycling Rate Increase",
    "sust_waste_reduction": "Waste Reduction",
    "sust_co2_reduction": "CO₂ Emission Reduction",
}

TARGET_SHORT = {
    "sust_recycling_rate": "recycling",
    "sust_waste_reduction": "waste",
    "sust_co2_reduction": "co2",
}

TARGET_ICONS = {
    "sust_recycling_rate": "♻️",
    "sust_waste_reduction": "🗑️",
    "sust_co2_reduction": "🌍",
}

COUNTRY_FLAGS = {
    "Slovakia": "🇸🇰",
    "Croatia": "🇭🇷",
    "Slovenia": "🇸🇮",
}

PROJECT_SIZE_OPTIONS = ["Small (<€180k)", "Medium (€180k–€5.35M)", "Large (>€5.35M)"]
COMPANY_SIZE_OPTIONS = ["Small (<50)", "Medium (50–249)", "Large (250+)"]
PROJECT_SIZE_MAP = {"Small (<€180k)": 0, "Medium (€180k–€5.35M)": 1, "Large (>€5.35M)": 2}
COMPANY_SIZE_MAP = {"Small (<50)": 0, "Medium (50–249)": 1, "Large (250+)": 2}
