"""Global configuration for the BIM-LCA-Circularity project."""
from pathlib import Path

# Reproducibility
RANDOM_SEED = 42

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw.xlsx"
DATA_CLEAN = PROJECT_ROOT / "dataset_clean.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"

for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Column name mapping (original Excel col index -> clean snake_case name)
COLUMN_NAMES = {
    1: "country",
    2: "countries_operated_in",
    3: "participant_status",
    4: "main_construction_activity",
    5: "project_size",
    6: "company_size",
    7: "foreign_participation",
    8: "legal_form",
    9: "project_type",
    10: "company_name",  # to be dropped
    11: "bim_feasibility_study",
    12: "bim_energy_sustainability",
    13: "bim_construction_mgmt",
    14: "bim_space_tracking",
    15: "bim_demolition",
    16: "bim_recycling",
    17: "sust_recycling_rate",
    18: "sust_waste_reduction",
    19: "sust_co2_reduction",
}

# Feature groups
BIM_USE_COLS = [
    "bim_feasibility_study",
    "bim_energy_sustainability",
    "bim_construction_mgmt",
    "bim_space_tracking",
    "bim_demolition",
    "bim_recycling",
]

TARGET_COLS = [
    "sust_recycling_rate",
    "sust_waste_reduction",
    "sust_co2_reduction",
]

SUSTAINABILITY_COLS = TARGET_COLS  # alias

CATEGORICAL_FEATURES = [
    "country",
    "participant_status",
    "main_construction_activity",
    "project_size",
    "company_size",
    "foreign_participation",
    "legal_form",
    "project_type",
]

ORDINAL_FEATURES = BIM_USE_COLS

# Ordinal mappings for encoding
PROJECT_SIZE_ORDER = {"Small": 0, "Medium": 1, "Large": 2}
COMPANY_SIZE_ORDER = {"Small": 0, "Medium": 1, "Large": 2}

# Plotting
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"
COLOR_PALETTE = "colorblind"
FIGSIZE_SINGLE = (8, 6)
FIGSIZE_WIDE = (12, 6)
FIGSIZE_LARGE = (14, 10)

# Cross-validation
CV_FOLDS = 5
CV_REPEATS = 5
