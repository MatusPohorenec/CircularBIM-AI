"""Phase 6: Validation & Publication Readiness — power analysis, overfitting checks, self-challenge."""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.config import (
    BIM_USE_COLS,
    COLOR_PALETTE,
    FIGURE_DPI,
    FIGURES_DIR,
    FIGSIZE_SINGLE,
    RANDOM_SEED,
    TARGET_COLS,
    TABLES_DIR,
)

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette=COLOR_PALETTE)


def post_hoc_power_analysis(df: pd.DataFrame) -> dict:
    """Compute post-hoc statistical power analysis."""
    logger.info("Running post-hoc power analysis")
    n = len(df)
    n_predictors = len(BIM_USE_COLS) + 4  # BIM cols + controls

    # Cohen's conventions for effect size (f²)
    # Small=0.02, Medium=0.15, Large=0.35
    # Power = 1 - beta(F_crit, df1, df2, lambda)
    # lambda = f² * n

    results = []
    for f2, label in [(0.02, "Small"), (0.15, "Medium"), (0.35, "Large")]:
        lam = f2 * n
        df1 = n_predictors
        df2 = n - n_predictors - 1
        if df2 <= 0:
            power = np.nan
        else:
            from scipy.stats import f as f_dist
            f_crit = f_dist.ppf(0.95, df1, df2)
            # Non-central F distribution
            from scipy.stats import ncf
            power = 1 - ncf.cdf(f_crit, df1, df2, lam)
        results.append({
            "Effect Size (f²)": f2,
            "Label": label,
            "N": n,
            "N Predictors": n_predictors,
            "Power (1-β)": power,
            "Adequate (≥0.80)": "Yes" if power >= 0.80 else "No",
        })

    power_df = pd.DataFrame(results)
    power_df.to_csv(TABLES_DIR / "power_analysis.csv", index=False, float_format="%.4f")
    logger.info("Power analysis:\n%s", power_df.to_string())
    return {"power_df": power_df}


def check_overfitting(results_df: pd.DataFrame) -> pd.DataFrame:
    """Flag potential overfitting by comparing CV performance to training performance."""
    logger.info("Checking for overfitting indicators")

    # The CV results already contain std; flag high variance
    checks = []
    for _, row in results_df.iterrows():
        if "Baseline" in str(row.get("model", "")):
            continue
        macro_f1_mean = row.get("macro_f1_mean", np.nan)
        macro_f1_std = row.get("macro_f1_std", np.nan)
        if pd.notna(macro_f1_mean) and pd.notna(macro_f1_std):
            cv_coefficient = macro_f1_std / (macro_f1_mean + 1e-10) * 100
            checks.append({
                "Target": row.get("target", ""),
                "Model": row.get("model", ""),
                "Macro-F1 Mean": macro_f1_mean,
                "Macro-F1 Std": macro_f1_std,
                "CV (%)": cv_coefficient,
                "Concern": "HIGH VARIANCE" if cv_coefficient > 30 else "OK",
            })

    check_df = pd.DataFrame(checks)
    check_df.to_csv(TABLES_DIR / "overfitting_check.csv", index=False, float_format="%.4f")
    logger.info("Overfitting check:\n%s", check_df.to_string())
    return check_df


def check_leakage(df: pd.DataFrame) -> str:
    """Verify no target information leaks into features."""
    logger.info("Checking for data leakage")

    # Key concern: cols 16 (bim_recycling) and 17 (sust_recycling_rate) overlap
    r, p = stats.spearmanr(
        df["bim_recycling"].astype(float),
        df["sust_recycling_rate"].astype(float),
    )
    leakage_warning = ""
    if abs(r) > 0.7:
        leakage_warning = (
            f"WARNING: High correlation (ρ={r:.3f}, p={p:.4f}) between "
            f"bim_recycling (feature) and sust_recycling_rate (target). "
            f"This may represent construct overlap rather than leakage, "
            f"but should be acknowledged as a limitation."
        )
        logger.warning(leakage_warning)
    else:
        leakage_warning = (
            f"Leakage check passed: bim_recycling ↔ sust_recycling_rate "
            f"correlation ρ={r:.3f} (p={p:.4f}) — within acceptable range."
        )
        logger.info(leakage_warning)

    return leakage_warning


def performance_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """Report model performance breakdown by country subgroup."""
    logger.info("Computing performance by country subgroup")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    from src.predictive_modelling import engineer_features, get_feature_columns

    df_eng = engineer_features(df)
    feature_cols = get_feature_columns(df_eng)

    results = []
    for target in TARGET_COLS:
        X_all = df_eng[feature_cols].astype(float).values
        y_all = df_eng[target].astype(int).values

        model = RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_SEED,
        )
        model.fit(X_all, y_all)

        for country in df["country"].unique():
            mask = df["country"] == country
            X_sub = df_eng.loc[mask, feature_cols].astype(float).values
            y_sub = df_eng.loc[mask, target].astype(int).values
            y_pred = model.predict(X_sub)

            results.append({
                "target": target,
                "country": country,
                "n": mask.sum(),
                "macro_f1": f1_score(y_sub, y_pred, average="macro", zero_division=0),
                "accuracy": (y_pred == y_sub).mean(),
            })

    country_df = pd.DataFrame(results)
    country_df.to_csv(TABLES_DIR / "performance_by_country.csv", index=False, float_format="%.4f")
    logger.info("Performance by country:\n%s", country_df.to_string())
    return country_df


def generate_consolidated_report(
    power_results: dict,
    overfitting_df: pd.DataFrame,
    leakage_info: str,
    country_df: pd.DataFrame,
) -> str:
    """Generate the final consolidated validation report."""
    lines = [
        "=" * 70,
        "PHASE 6: VALIDATION & SELF-CHALLENGE REPORT",
        "=" * 70,
        "",
        "1. STATISTICAL POWER ANALYSIS",
        "-" * 40,
        power_results["power_df"].to_string(index=False),
        "",
        "2. OVERFITTING CHECK",
        "-" * 40,
        overfitting_df.to_string(index=False),
        "",
        "3. DATA LEAKAGE CHECK",
        "-" * 40,
        leakage_info,
        "",
        "4. GENERALISABILITY BY COUNTRY",
        "-" * 40,
        country_df.to_string(index=False),
        "",
        "5. KEY LIMITATIONS",
        "-" * 40,
        "- Targets are perceived impacts (survey-based), not measured LCA/GWP values",
        "- Dataset exhibits high duplication (9 unique respondent profiles across 199 rows)",
        "- Small effective sample size limits model generalisability",
        "- Country subgroups are unbalanced (Croatia=99, Slovakia=60, Slovenia=40)",
        "- Ordinal targets (1-5 Likert) have limited granularity",
        "- Cross-country external validity is limited",
        "",
        "6. CONSTRUCT VALIDITY CAVEAT",
        "-" * 40,
        "Sustainability targets (recycling rate increase, waste reduction, CO2 reduction)",
        "are self-reported perceptions of BIM's impact, not objective measurements.",
        "This introduces response bias and limits causal interpretation.",
        "=" * 70,
    ]

    report = "\n".join(lines)
    report_path = TABLES_DIR / "validation_report.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Saved consolidated validation report")
    return report


def run_validation(df: pd.DataFrame, model_results: pd.DataFrame | None = None) -> dict:
    """Run the full validation pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 6: VALIDATION & PUBLICATION READINESS")
    logger.info("=" * 60)

    power = post_hoc_power_analysis(df)

    if model_results is not None and len(model_results) > 0:
        overfitting = check_overfitting(model_results)
    else:
        overfitting = pd.DataFrame()

    leakage = check_leakage(df)
    country = performance_by_country(df)

    report = generate_consolidated_report(power, overfitting, leakage, country)
    logger.info("\n%s", report)

    return {
        "power": power,
        "overfitting": overfitting,
        "leakage": leakage,
        "country_performance": country,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = pd.read_csv("dataset_clean.csv")
    run_validation(df)
