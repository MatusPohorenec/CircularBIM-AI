"""Model loading and inference for the Streamlit app."""
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from src.st_config import (
    MODELS_DIR,
    TARGET_SHORT,
    COMPANY_SIZE_MAP,
    PROJECT_SIZE_MAP,
)

BIM_COLS = [
    "bim_feasibility_study",
    "bim_energy_sustainability",
    "bim_construction_mgmt",
    "bim_space_tracking",
    "bim_demolition",
    "bim_recycling",
]


@st.cache_resource
def load_model(target_key: str):
    """Load a serialised model by target short name."""
    short = TARGET_SHORT[target_key]
    return joblib.load(MODELS_DIR / f"model_{short}.joblib")


@st.cache_resource
def load_feature_columns() -> list[str]:
    return joblib.load(MODELS_DIR / "feature_columns.joblib")


@st.cache_resource
def load_label_encoders() -> dict:
    return joblib.load(MODELS_DIR / "label_encoders.joblib")


def build_feature_vector(
    bim_values: dict[str, int],
    country: str,
    participant_status: str,
    project_size: str,
    company_size: str,
    foreign_participation: bool,
    countries_operated_in: int,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame matching training schema."""
    feat_cols = load_feature_columns()

    row = {}
    for col in BIM_COLS:
        row[col] = float(bim_values.get(col, 1))

    bim_vals = np.array([row[c] for c in BIM_COLS])
    row["bim_maturity_index"] = float(bim_vals.mean())
    row["bim_lifecycle_coverage"] = float((bim_vals >= 3).sum())
    row["bim_eol_focus"] = float(np.mean([row["bim_demolition"], row["bim_recycling"]]))

    row["project_size_ord"] = float(PROJECT_SIZE_MAP.get(project_size, 1))
    row["company_size_ord"] = float(COMPANY_SIZE_MAP.get(company_size, 1))

    row["company_x_bim_maturity"] = row["company_size_ord"] * row["bim_maturity_index"]
    row["project_x_bim_eol"] = row["project_size_ord"] * row["bim_eol_focus"]

    row["countries_operated_in"] = float(countries_operated_in)
    row["foreign_participation_bin"] = 1.0 if foreign_participation else 0.0

    # Country dummies (training used drop_first=True, reference=Croatia)
    row["country_Slovakia"] = 1.0 if country == "Slovakia" else 0.0
    row["country_Slovenia"] = 1.0 if country == "Slovenia" else 0.0

    # Status dummies (training used drop_first=True)
    # Need to check what was the reference category
    row["status_Designer"] = 1.0 if participant_status == "Designer" else 0.0
    row["status_Investor"] = 1.0 if participant_status == "Investor" else 0.0

    # Build DataFrame with exact column order
    df = pd.DataFrame([{col: row.get(col, 0.0) for col in feat_cols}])
    return df


def predict(feature_df: pd.DataFrame, target_key: str) -> dict:
    """Run prediction and return class + probabilities."""
    model = load_model(target_key)
    X = feature_df.astype(float).values

    pred_class = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    classes = model.classes_

    proba_dict = {int(c): float(p) for c, p in zip(classes, proba)}

    # Expected value
    expected = sum(c * p for c, p in proba_dict.items())

    return {
        "prediction": pred_class,
        "probabilities": proba_dict,
        "expected": round(expected, 2),
    }


def predict_all(feature_df: pd.DataFrame) -> dict[str, dict]:
    """Predict all three targets."""
    from src.st_config import TARGET_LABELS
    results = {}
    for target_key in TARGET_LABELS:
        results[target_key] = predict(feature_df, target_key)
    return results
