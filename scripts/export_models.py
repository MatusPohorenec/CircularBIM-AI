"""Export trained models and pre-compute aggregate data for the Streamlit frontend.

Outputs:
  models/*.joblib          — serialised ML models + encoders
  assets/eda_stats.json    — aggregate descriptive statistics
  assets/distributions.json — class distributions per target/country
  assets/correlations.json — Spearman correlation matrix
  assets/bim_heatmap.json  — mean BIM-use by country/phase
  assets/shap_data.json    — mean |SHAP| per feature per target
  assets/model_results.json — model comparison table
  assets/mcdm_results.json — scenario rankings + sensitivity
"""
import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BIM_USE_COLS,
    CATEGORICAL_FEATURES,
    COMPANY_SIZE_ORDER,
    DATA_CLEAN,
    PROJECT_SIZE_ORDER,
    RANDOM_SEED,
    TARGET_COLS,
)

MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

np.random.seed(RANDOM_SEED)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CLEAN)
    logger.info("Loaded %d rows from %s", len(df), DATA_CLEAN)
    return df


def export_label_encoders(df: pd.DataFrame) -> dict:
    """Fit and export label encoders for categorical features."""
    encoders = {}
    cat_cols = ["country", "participant_status", "main_construction_activity",
                "legal_form", "project_type"]
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    joblib.dump(encoders, MODELS_DIR / "label_encoders.joblib")
    logger.info("Exported label encoders")
    return encoders


def engineer_features_for_export(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Replicate feature engineering from predictive_modelling.py."""
    data = df.copy()
    bim_vals = data[BIM_USE_COLS].astype(float)
    data["bim_maturity_index"] = bim_vals.mean(axis=1)
    data["bim_lifecycle_coverage"] = (bim_vals >= 3).sum(axis=1)
    data["bim_eol_focus"] = bim_vals[["bim_demolition", "bim_recycling"]].mean(axis=1)
    data["project_size_ord"] = data["project_size"].map(PROJECT_SIZE_ORDER).fillna(1)
    data["company_size_ord"] = data["company_size"].map(COMPANY_SIZE_ORDER).fillna(1)
    data["company_x_bim_maturity"] = data["company_size_ord"] * data["bim_maturity_index"]
    data["project_x_bim_eol"] = data["project_size_ord"] * data["bim_eol_focus"]

    country_dummies = pd.get_dummies(data["country"], prefix="country", drop_first=True, dtype=float)
    data = pd.concat([data, country_dummies], axis=1)

    status_dummies = pd.get_dummies(data["participant_status"], prefix="status", drop_first=True, dtype=float)
    data = pd.concat([data, status_dummies], axis=1)

    data["foreign_participation_bin"] = (data["foreign_participation"] == "Yes").astype(float)

    feature_cols = (
        BIM_USE_COLS
        + ["bim_maturity_index", "bim_lifecycle_coverage", "bim_eol_focus"]
        + ["project_size_ord", "company_size_ord"]
        + ["company_x_bim_maturity", "project_x_bim_eol"]
        + ["countries_operated_in", "foreign_participation_bin"]
        + [c for c in data.columns if c.startswith("country_")]
        + [c for c in data.columns if c.startswith("status_")]
    )
    feature_cols = [c for c in feature_cols if c in data.columns]
    return data, feature_cols


def train_and_export_models(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Train final models for all model types and export."""
    from sklearn.svm import SVC
    import xgboost as xgb
    import lightgbm as lgb

    df_eng, feat_cols = engineer_features_for_export(df)

    target_names = {
        "sust_recycling_rate": "recycling",
        "sust_waste_reduction": "waste",
        "sust_co2_reduction": "co2",
    }

    model_defs = {
        "rf": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "xgb": lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, eval_metric="mlogloss", verbosity=0,
            device="cuda", tree_method="hist",
        ),
        "lgb": lambda: lgb.LGBMClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=RANDOM_SEED, verbose=-1,
            device="gpu",
        ),
        "svm": lambda: SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
            probability=True, random_state=RANDOM_SEED,
        ),
    }

    for target, short_name in target_names.items():
        X = df_eng[feat_cols].astype(float).values
        y = df_eng[target].astype(int).values

        for model_type, model_factory in model_defs.items():
            model = model_factory()

            # XGBoost needs 0-indexed labels
            if model_type == "xgb":
                classes = np.sort(np.unique(y))
                label_map = {c: i for i, c in enumerate(classes)}
                y_fit = np.array([label_map[v] for v in y])
                model.fit(X, y_fit)
                # Store original classes for inverse mapping
                model._original_classes = classes
            else:
                model.fit(X, y)

            # Save with model type suffix (and keep backward-compat RF alias)
            model_path = MODELS_DIR / f"model_{short_name}_{model_type}.joblib"
            joblib.dump(model, model_path)
            logger.info("Exported %s/%s -> %s", target, model_type, model_path)

            # Backward compatibility: also save RF as the default model
            if model_type == "rf":
                compat_path = MODELS_DIR / f"model_{short_name}.joblib"
                joblib.dump(model, compat_path)

    # Save feature columns list
    joblib.dump(feat_cols, MODELS_DIR / "feature_columns.joblib")
    logger.info("Exported feature_columns.joblib")


def precompute_eda_stats(df: pd.DataFrame) -> None:
    """Pre-compute aggregate EDA statistics."""
    numeric_cols = BIM_USE_COLS + TARGET_COLS + ["countries_operated_in"]

    # Descriptive stats
    desc = df[numeric_cols].describe().round(3)
    desc.loc["median"] = df[numeric_cols].median().round(3)
    eda_stats = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "countries": df["country"].value_counts().to_dict(),
        "descriptive": {col: desc[col].to_dict() for col in numeric_cols},
    }

    # Categorical distributions
    cat_dists = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            cat_dists[col] = df[col].value_counts().to_dict()
    eda_stats["categorical"] = cat_dists

    with open(ASSETS_DIR / "eda_stats.json", "w") as f:
        json.dump(eda_stats, f, indent=2, default=str)
    logger.info("Exported eda_stats.json")


def precompute_distributions(df: pd.DataFrame) -> None:
    """Pre-compute target distributions by country."""
    distributions = {}
    for target in TARGET_COLS:
        dist = {}
        for country in df["country"].unique():
            vc = df[df["country"] == country][target].value_counts().sort_index()
            dist[country] = {int(k): int(v) for k, v in vc.items()}
        # Overall
        vc_all = df[target].value_counts().sort_index()
        dist["All"] = {int(k): int(v) for k, v in vc_all.items()}
        distributions[target] = dist

    with open(ASSETS_DIR / "distributions.json", "w") as f:
        json.dump(distributions, f, indent=2)
    logger.info("Exported distributions.json")


def precompute_correlations(df: pd.DataFrame) -> None:
    """Pre-compute Spearman correlation matrix."""
    ordinal_cols = BIM_USE_COLS + TARGET_COLS
    corr = df[ordinal_cols].astype(float).corr(method="spearman").round(4)
    corr_dict = {
        "columns": ordinal_cols,
        "values": corr.values.tolist(),
    }
    with open(ASSETS_DIR / "correlations.json", "w") as f:
        json.dump(corr_dict, f, indent=2)
    logger.info("Exported correlations.json")


def precompute_bim_heatmap(df: pd.DataFrame) -> None:
    """Pre-compute mean BIM-use by country and phase."""
    bim_means = df.groupby("country")[BIM_USE_COLS].mean().round(3)
    heatmap = {
        "countries": bim_means.index.tolist(),
        "phases": BIM_USE_COLS,
        "values": bim_means.values.tolist(),
        "phase_labels": [
            "Feasibility Study",
            "Energy Sustainability",
            "Construction Mgmt",
            "Space Tracking",
            "Demolition",
            "Recycling",
        ],
    }
    with open(ASSETS_DIR / "bim_heatmap.json", "w") as f:
        json.dump(heatmap, f, indent=2)
    logger.info("Exported bim_heatmap.json")


def precompute_shap_data(df: pd.DataFrame) -> None:
    """Pre-compute mean |SHAP| values per feature per target."""
    import shap

    df_eng, feat_cols = engineer_features_for_export(df)
    shap_data = {}

    for target in TARGET_COLS:
        X = df_eng[feat_cols].astype(float)
        y = df_eng[target].astype(int).values

        model = RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
        )
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)

        if isinstance(sv, list):
            mean_abs = np.mean([np.abs(s) for s in sv], axis=0).mean(axis=0)
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            mean_abs = np.abs(sv).mean(axis=2).mean(axis=0)
        else:
            mean_abs = np.abs(sv).mean(axis=0)

        importance = sorted(
            zip(feat_cols, mean_abs.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        shap_data[target] = [{"feature": f, "importance": round(v, 6)} for f, v in importance]

    with open(ASSETS_DIR / "shap_data.json", "w") as f:
        json.dump(shap_data, f, indent=2)
    logger.info("Exported shap_data.json")


def precompute_model_results() -> None:
    """Copy model comparison results to assets."""
    results_path = PROJECT_ROOT / "results" / "tables" / "model_comparison.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        records = df.to_dict(orient="records")
        # Clean NaN values
        for r in records:
            for k, v in r.items():
                if isinstance(v, float) and np.isnan(v):
                    r[k] = None
        with open(ASSETS_DIR / "model_results.json", "w") as f:
            json.dump(records, f, indent=2)
        logger.info("Exported model_results.json")
    else:
        logger.warning("model_comparison.csv not found; skipping")


def precompute_hyperparameter_data() -> None:
    """Export hyperparameter configs and grid search results to assets."""
    # Hyperparameter configs
    config_path = PROJECT_ROOT / "results" / "tables" / "hyperparameter_configs.csv"
    search_path = PROJECT_ROOT / "results" / "tables" / "hyperparameter_search.csv"

    hp_data = {"configs": [], "search_results": [], "grids": {}}

    if config_path.exists():
        df = pd.read_csv(config_path)
        hp_data["configs"] = df.to_dict(orient="records")

    if search_path.exists():
        df = pd.read_csv(search_path)
        records = df.to_dict(orient="records")
        for r in records:
            for k, v in r.items():
                if isinstance(v, float) and np.isnan(v):
                    r[k] = None
        hp_data["search_results"] = records

    # Include the grid definitions
    from src.predictive_modelling import get_hyperparameter_grids
    grids = get_hyperparameter_grids()
    hp_data["grids"] = {k: {pk: [str(x) for x in pv] for pk, pv in v.items()}
                        for k, v in grids.items()}

    with open(ASSETS_DIR / "hyperparameters.json", "w") as f:
        json.dump(hp_data, f, indent=2, default=str)
    logger.info("Exported hyperparameters.json")


def precompute_cv_details() -> None:
    """Export per-fold CV results to assets."""
    fold_path = PROJECT_ROOT / "results" / "tables" / "cv_fold_results.csv"
    if fold_path.exists():
        df = pd.read_csv(fold_path)
        records = df.to_dict(orient="records")
        for r in records:
            for k, v in r.items():
                if isinstance(v, float) and np.isnan(v):
                    r[k] = None
        with open(ASSETS_DIR / "cv_details.json", "w") as f:
            json.dump(records, f, indent=2)
        logger.info("Exported cv_details.json (%d fold records)", len(records))
    else:
        logger.warning("cv_fold_results.csv not found; skipping")


def precompute_mcdm_results() -> None:
    """Copy MCDM results to assets."""
    ranking_path = PROJECT_ROOT / "results" / "tables" / "mcdm_ranking.csv"
    sens_path = PROJECT_ROOT / "results" / "tables" / "mcdm_sensitivity.csv"
    pred_path = PROJECT_ROOT / "results" / "tables" / "scenario_predictions.csv"

    mcdm = {}
    if ranking_path.exists():
        mcdm["ranking"] = pd.read_csv(ranking_path).to_dict(orient="records")
    if sens_path.exists():
        mcdm["sensitivity"] = pd.read_csv(sens_path).to_dict(orient="records")
    if pred_path.exists():
        mcdm["predictions"] = pd.read_csv(pred_path).to_dict(orient="records")

    with open(ASSETS_DIR / "mcdm_results.json", "w") as f:
        json.dump(mcdm, f, indent=2, default=str)
    logger.info("Exported mcdm_results.json")


def precompute_reliability() -> None:
    """Copy reliability results to assets."""
    rel_path = PROJECT_ROOT / "results" / "tables" / "reliability_results.csv"
    if rel_path.exists():
        df = pd.read_csv(rel_path)
        with open(ASSETS_DIR / "reliability.json", "w") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2)
        logger.info("Exported reliability.json")


def precompute_sem_results() -> None:
    """Copy SEM path coefficients to assets."""
    sem_path = PROJECT_ROOT / "results" / "tables" / "pls_sem_estimates.csv"
    fit_path = PROJECT_ROOT / "results" / "tables" / "pls_sem_fit_stats.csv"
    sem_data = {}
    if sem_path.exists():
        df = pd.read_csv(sem_path)
        records = df.to_dict(orient="records")
        for r in records:
            for k, v in r.items():
                if isinstance(v, float) and np.isnan(v):
                    r[k] = None
        sem_data["estimates"] = records
    if fit_path.exists():
        df = pd.read_csv(fit_path)
        records = df.to_dict(orient="records")
        for r in records:
            for k, v in r.items():
                if isinstance(v, float) and np.isnan(v):
                    r[k] = None
        sem_data["fit"] = records
    with open(ASSETS_DIR / "sem_results.json", "w") as f:
        json.dump(sem_data, f, indent=2)
    logger.info("Exported sem_results.json")


def main() -> None:
    logger.info("=" * 60)
    logger.info("EXPORTING MODELS & PRE-COMPUTING FRONTEND ASSETS")
    logger.info("=" * 60)

    df = load_data()

    # Export models and encoders
    export_label_encoders(df)
    df_eng, feat_cols = engineer_features_for_export(df)
    train_and_export_models(df, feat_cols)

    # Pre-compute all aggregate data
    precompute_eda_stats(df)
    precompute_distributions(df)
    precompute_correlations(df)
    precompute_bim_heatmap(df)
    precompute_shap_data(df)
    precompute_model_results()
    precompute_hyperparameter_data()
    precompute_cv_details()
    precompute_mcdm_results()
    precompute_reliability()
    precompute_sem_results()

    logger.info("=" * 60)
    logger.info("EXPORT COMPLETE")
    logger.info("Models: %s", list(MODELS_DIR.glob("*.joblib")))
    logger.info("Assets: %s", list(ASSETS_DIR.glob("*.json")))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
