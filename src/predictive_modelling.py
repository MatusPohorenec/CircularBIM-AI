"""Phase 4: Predictive AI Modelling — ML models with ordinal classification, SHAP analysis."""
import logging
import warnings
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from src.config import (
    BIM_USE_COLS,
    CATEGORICAL_FEATURES,
    COLOR_PALETTE,
    COMPANY_SIZE_ORDER,
    CV_FOLDS,
    CV_REPEATS,
    FIGURE_DPI,
    FIGURES_DIR,
    FIGSIZE_LARGE,
    FIGSIZE_SINGLE,
    FIGSIZE_WIDE,
    MODELS_DIR,
    PROJECT_SIZE_ORDER,
    RANDOM_SEED,
    TARGET_COLS,
    TABLES_DIR,
)

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette=COLOR_PALETTE)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features for ML modelling."""
    logger.info("Engineering features")
    data = df.copy()

    # BIM composite features
    bim_vals = data[BIM_USE_COLS].astype(float)
    data["bim_maturity_index"] = bim_vals.mean(axis=1)
    data["bim_lifecycle_coverage"] = (bim_vals >= 3).sum(axis=1)
    data["bim_eol_focus"] = bim_vals[["bim_demolition", "bim_recycling"]].mean(axis=1)

    # Ordinal encoding for sizes
    data["project_size_ord"] = data["project_size"].map(PROJECT_SIZE_ORDER).fillna(1)
    data["company_size_ord"] = data["company_size"].map(COMPANY_SIZE_ORDER).fillna(1)

    # Interaction features
    data["company_x_bim_maturity"] = data["company_size_ord"] * data["bim_maturity_index"]
    data["project_x_bim_eol"] = data["project_size_ord"] * data["bim_eol_focus"]

    # Country encoding (ordinal or one-hot)
    country_dummies = pd.get_dummies(data["country"], prefix="country", drop_first=True, dtype=float)
    data = pd.concat([data, country_dummies], axis=1)

    # Participant status encoding
    status_dummies = pd.get_dummies(data["participant_status"], prefix="status", drop_first=True, dtype=float)
    data = pd.concat([data, status_dummies], axis=1)

    # Foreign participation binary
    data["foreign_participation_bin"] = (data["foreign_participation"] == "Yes").astype(float)

    logger.info("Feature engineering complete. New columns added.")
    return data


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get the list of feature columns for ML models."""
    feature_cols = (
        BIM_USE_COLS
        + ["bim_maturity_index", "bim_lifecycle_coverage", "bim_eol_focus"]
        + ["project_size_ord", "company_size_ord"]
        + ["company_x_bim_maturity", "project_x_bim_eol"]
        + ["countries_operated_in", "foreign_participation_bin"]
        + [c for c in df.columns if c.startswith("country_")]
        + [c for c in df.columns if c.startswith("status_")]
    )
    return [c for c in feature_cols if c in df.columns]


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    classes: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute evaluation metrics for ordinal classification."""
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "cohen_kappa_qw": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mae": mean_absolute_error(y_true, y_pred),
    }

    # AUC-ROC (one-vs-rest) if probabilities available
    if y_proba is not None and classes is not None and len(classes) > 2:
        try:
            metrics["auc_roc_ovr"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro",
                labels=classes,
            )
        except Exception:
            metrics["auc_roc_ovr"] = np.nan
    else:
        metrics["auc_roc_ovr"] = np.nan

    return metrics


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_name: str,
    n_splits: int = CV_FOLDS,
    n_repeats: int = CV_REPEATS,
) -> dict:
    """Run repeated stratified k-fold cross-validation."""
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_SEED
    )
    classes = np.sort(np.unique(y))

    # For XGBoost: remap labels to 0-indexed
    import xgboost as xgb
    needs_remap = isinstance(model, xgb.XGBClassifier)
    if needs_remap:
        label_map = {c: i for i, c in enumerate(classes)}
        inv_label_map = {i: c for c, i in label_map.items()}

    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            model_clone = _clone_model(model)
            if needs_remap:
                y_train_fit = np.array([label_map[v] for v in y_train])
                model_clone.fit(X_train, y_train_fit)
                y_pred_raw = model_clone.predict(X_test)
                y_pred = np.array([inv_label_map[v] for v in y_pred_raw])
            else:
                model_clone.fit(X_train, y_train)
                y_pred = model_clone.predict(X_test)

            # Get probabilities if available
            y_proba = None
            if hasattr(model_clone, "predict_proba"):
                try:
                    y_proba = model_clone.predict_proba(X_test)
                    # Ensure proba has correct shape
                    if y_proba.shape[1] != len(classes):
                        y_proba = None
                except Exception:
                    y_proba = None

            fold_metric = evaluate_model(y_test, y_pred, y_proba, classes)
            fold_metrics.append(fold_metric)
        except Exception as e:
            logger.warning("  Fold %d failed for %s: %s", fold_idx, model_name, e)

    if not fold_metrics:
        return {"error": "All folds failed"}

    # Aggregate metrics
    metric_names = fold_metrics[0].keys()
    agg = {}
    for m in metric_names:
        values = [fm[m] for fm in fold_metrics if not np.isnan(fm[m])]
        if values:
            agg[f"{m}_mean"] = np.mean(values)
            agg[f"{m}_std"] = np.std(values)
            agg[f"{m}_ci_low"] = np.percentile(values, 2.5)
            agg[f"{m}_ci_high"] = np.percentile(values, 97.5)
        else:
            agg[f"{m}_mean"] = np.nan
            agg[f"{m}_std"] = np.nan

    agg["n_folds_completed"] = len(fold_metrics)
    return agg


def _clone_model(model: Any) -> Any:
    """Clone a model instance."""
    from sklearn.base import clone
    try:
        return clone(model)
    except Exception:
        # Fallback: re-instantiate
        return model.__class__(**model.get_params())


def build_models() -> dict[str, Any]:
    """Build the model zoo for comparison."""
    from sklearn.svm import SVC
    import xgboost as xgb
    import lightgbm as lgb

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            eval_metric="mlogloss",
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            verbose=-1,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_SEED,
        ),
    }

    # Try mord for ordinal-specific models
    try:
        import mord
        models["Ordinal Ridge (mord)"] = mord.OrdinalRidge(alpha=1.0)
        models["LAD (mord)"] = mord.LAD(C=1.0)
    except Exception as e:
        logger.warning("mord models not available: %s", e)

    return models


def compute_baseline_metrics(y: np.ndarray) -> dict:
    """Compute majority class and random baseline metrics."""
    classes, counts = np.unique(y, return_counts=True)
    majority_class = classes[np.argmax(counts)]

    # Majority baseline
    y_pred_majority = np.full_like(y, majority_class)
    majority_metrics = evaluate_model(y, y_pred_majority)

    # Random baseline
    rng = np.random.RandomState(RANDOM_SEED)
    y_pred_random = rng.choice(classes, size=len(y))
    random_metrics = evaluate_model(y, y_pred_random)

    return {
        "majority_class": majority_metrics,
        "random": random_metrics,
    }


def run_model_comparison(df: pd.DataFrame) -> dict:
    """Run all models on all targets with cross-validation."""
    logger.info("Running model comparison across all targets")

    feature_cols = get_feature_columns(df)
    models = build_models()

    all_results = []
    for target in TARGET_COLS:
        logger.info("--- Target: %s ---", target)

        X = df[feature_cols].astype(float).values
        y = df[target].astype(int).values

        # Baseline
        baselines = compute_baseline_metrics(y)
        for bname, bmetrics in baselines.items():
            row = {"target": target, "model": f"Baseline ({bname})"}
            for k, v in bmetrics.items():
                row[f"{k}_mean"] = v
                row[f"{k}_std"] = 0.0
            all_results.append(row)

        # Models
        for model_name, model in models.items():
            logger.info("  Model: %s", model_name)
            cv_result = cross_validate_model(
                model, X, y, feature_cols, model_name
            )
            row = {"target": target, "model": model_name}
            row.update(cv_result)
            all_results.append(row)
            logger.info("    Macro-F1: %.4f ± %.4f, κ: %.4f ± %.4f",
                        cv_result.get("macro_f1_mean", 0),
                        cv_result.get("macro_f1_std", 0),
                        cv_result.get("cohen_kappa_qw_mean", 0),
                        cv_result.get("cohen_kappa_qw_std", 0))

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(TABLES_DIR / "model_comparison.csv", index=False, float_format="%.4f")
    logger.info("Saved model comparison table")

    return {"results_df": results_df, "feature_cols": feature_cols}


def run_shap_analysis(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Run SHAP analysis for the best model (Random Forest) on all targets."""
    logger.info("Running SHAP analysis")
    import shap

    for target in TARGET_COLS:
        logger.info("  SHAP for target: %s", target)
        X = df[feature_cols].astype(float)
        y = df[target].astype(int).values

        # Train RF on full data for SHAP
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X, y)

        # SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Handle different shap_values formats
        if isinstance(shap_values, list):
            # List of arrays (one per class): average absolute values
            shap_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # 3D array (n_samples, n_features, n_classes): average over classes
            shap_mean = np.abs(shap_values).mean(axis=2)
        else:
            shap_mean = np.abs(shap_values)

        # Ensure 2D
        if shap_mean.ndim == 1:
            shap_mean = shap_mean.reshape(1, -1)

        # Feature importance
        mean_shap_per_feature = shap_mean.mean(axis=0)
        if len(mean_shap_per_feature) != len(feature_cols):
            logger.warning("SHAP shape mismatch: %d vs %d features", len(mean_shap_per_feature), len(feature_cols))
            continue

        importance = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": mean_shap_per_feature,
        }).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(TABLES_DIR / f"shap_importance_{target}.csv", index=False, float_format="%.4f")

        # SHAP bar plot
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        top_n = min(15, len(importance))
        top_features = importance.head(top_n)
        ax.barh(range(top_n), top_features["mean_abs_shap"].values, color=sns.color_palette(COLOR_PALETTE)[0])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features["feature"].values)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        target_label = target.replace("sust_", "").replace("_", " ").title()
        ax.set_title(f"SHAP Feature Importance — {target_label}")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"shap_bar_{target}.pdf", dpi=FIGURE_DPI, bbox_inches="tight")
        fig.savefig(FIGURES_DIR / f"shap_bar_{target}.png", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

        # SHAP beeswarm plot
        try:
            fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
            if isinstance(shap_values, list):
                sv_for_plot = shap_values[0] if len(shap_values) > 0 else shap_values
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                sv_for_plot = shap_values[:, :, 0]
            else:
                sv_for_plot = shap_values
            shap.summary_plot(
                sv_for_plot, X, feature_names=feature_cols,
                show=False, max_display=15,
            )
            plt.title(f"SHAP Summary — {target_label}")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"shap_beeswarm_{target}.pdf", dpi=FIGURE_DPI, bbox_inches="tight")
            plt.savefig(FIGURES_DIR / f"shap_beeswarm_{target}.png", dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close("all")
        except Exception as e:
            logger.warning("  Beeswarm plot failed for %s: %s", target, e)

        # Partial dependence for top 3 features
        top3 = importance.head(3)["feature"].tolist()
        for feat in top3:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                feat_idx = feature_cols.index(feat)
                if isinstance(shap_values, list):
                    sv_col = np.mean([sv[:, feat_idx] for sv in shap_values], axis=0)
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    sv_col = shap_values[:, feat_idx, :].mean(axis=1)
                else:
                    sv_col = shap_values[:, feat_idx]
                ax.scatter(X[feat].values, sv_col, alpha=0.5, s=20)
                ax.set_xlabel(feat)
                ax.set_ylabel("SHAP value")
                ax.set_title(f"SHAP Dependence: {feat} → {target_label}")
                fig.tight_layout()
                fig.savefig(FIGURES_DIR / f"shap_dep_{target}_{feat}.pdf", dpi=FIGURE_DPI, bbox_inches="tight")
                fig.savefig(FIGURES_DIR / f"shap_dep_{target}_{feat}.png", dpi=FIGURE_DPI, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                logger.warning("  Dependence plot failed for %s/%s: %s", target, feat, e)

    logger.info("SHAP analysis complete")


def run_predictive_modelling(df: pd.DataFrame) -> dict:
    """Run the full predictive modelling pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 4: PREDICTIVE AI MODELLING")
    logger.info("=" * 60)

    # Feature engineering
    df_eng = engineer_features(df)

    # Model comparison
    comparison = run_model_comparison(df_eng)

    # SHAP analysis
    run_shap_analysis(df_eng, comparison["feature_cols"])

    logger.info("Predictive modelling complete")
    return comparison


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    warnings.filterwarnings("ignore")
    np.random.seed(RANDOM_SEED)
    df = pd.read_csv("dataset_clean.csv")
    run_predictive_modelling(df)
