"""Phase 5: MCDM Layer — Scenario definition, prediction, ranking, sensitivity analysis."""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    BIM_USE_COLS,
    COLOR_PALETTE,
    FIGURE_DPI,
    FIGURES_DIR,
    FIGSIZE_LARGE,
    FIGSIZE_WIDE,
    RANDOM_SEED,
    TARGET_COLS,
    TABLES_DIR,
)

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette=COLOR_PALETTE)


def define_scenarios() -> pd.DataFrame:
    """Define BIM adoption scenarios varying across lifecycle phases."""
    scenarios = {
        "S1: Baseline (Low BIM)": {
            "bim_feasibility_study": 1,
            "bim_energy_sustainability": 1,
            "bim_construction_mgmt": 2,
            "bim_space_tracking": 1,
            "bim_demolition": 1,
            "bim_recycling": 1,
        },
        "S2: Design-Focused BIM": {
            "bim_feasibility_study": 4,
            "bim_energy_sustainability": 5,
            "bim_construction_mgmt": 3,
            "bim_space_tracking": 2,
            "bim_demolition": 1,
            "bim_recycling": 1,
        },
        "S3: Construction-Focused BIM": {
            "bim_feasibility_study": 2,
            "bim_energy_sustainability": 3,
            "bim_construction_mgmt": 5,
            "bim_space_tracking": 4,
            "bim_demolition": 1,
            "bim_recycling": 1,
        },
        "S4: End-of-Life Focused BIM": {
            "bim_feasibility_study": 2,
            "bim_energy_sustainability": 2,
            "bim_construction_mgmt": 3,
            "bim_space_tracking": 2,
            "bim_demolition": 5,
            "bim_recycling": 5,
        },
        "S5: Full Lifecycle BIM": {
            "bim_feasibility_study": 4,
            "bim_energy_sustainability": 5,
            "bim_construction_mgmt": 5,
            "bim_space_tracking": 4,
            "bim_demolition": 4,
            "bim_recycling": 4,
        },
        "S6: Maximum BIM Adoption": {
            "bim_feasibility_study": 5,
            "bim_energy_sustainability": 5,
            "bim_construction_mgmt": 5,
            "bim_space_tracking": 5,
            "bim_demolition": 5,
            "bim_recycling": 5,
        },
    }

    df = pd.DataFrame(scenarios).T
    df.index.name = "scenario"
    return df


def predict_scenario_outcomes(
    scenario_df: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Use trained ML model to predict sustainability outcomes for each scenario."""
    logger.info("Predicting scenario outcomes using trained RF model")

    from src.predictive_modelling import engineer_features, get_feature_columns

    # Prepare scenario data with default control values (median from training data)
    scenario_full = scenario_df.copy()

    # Add control variables with median/mode values from training
    scenario_full["countries_operated_in"] = train_df["countries_operated_in"].median()
    scenario_full["project_size"] = train_df["project_size"].mode()[0]
    scenario_full["company_size"] = train_df["company_size"].mode()[0]
    scenario_full["foreign_participation"] = train_df["foreign_participation"].mode()[0]
    scenario_full["participant_status"] = train_df["participant_status"].mode()[0]
    scenario_full["country"] = train_df["country"].mode()[0]
    scenario_full["legal_form"] = train_df["legal_form"].mode()[0]
    scenario_full["project_type"] = train_df["project_type"].mode()[0]
    scenario_full["main_construction_activity"] = train_df["main_construction_activity"].mode()[0]

    # Add dummy target columns for feature engineering
    for t in TARGET_COLS:
        scenario_full[t] = 3  # placeholder

    scenario_eng = engineer_features(scenario_full)
    eng_feature_cols = get_feature_columns(scenario_eng)

    # Align feature columns with training data
    train_eng = engineer_features(train_df)
    train_feature_cols = get_feature_columns(train_eng)

    # Use only features that exist in both
    common_features = [f for f in train_feature_cols if f in scenario_eng.columns]

    predictions = {}
    probas = {}
    for target in TARGET_COLS:
        X_train = train_eng[common_features].astype(float).values
        y_train = train_eng[target].astype(int).values
        X_scenario = scenario_eng[common_features].astype(float).values

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_scenario)
        proba = model.predict_proba(X_scenario)

        # Compute expected value (weighted average of class probabilities)
        classes = model.classes_
        expected = (proba * classes).sum(axis=1)

        predictions[target] = pred
        predictions[f"{target}_expected"] = expected
        probas[target] = proba

    pred_df = pd.DataFrame(predictions, index=scenario_df.index)
    pred_df.to_csv(TABLES_DIR / "scenario_predictions.csv", float_format="%.3f")
    logger.info("Scenario predictions:\n%s", pred_df.to_string())
    return pred_df


def topsis(decision_matrix: np.ndarray, weights: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    """TOPSIS method for MCDM ranking."""
    # Normalise decision matrix (vector normalisation)
    norms = np.sqrt((decision_matrix ** 2).sum(axis=0))
    norms[norms == 0] = 1  # avoid division by zero
    norm_matrix = decision_matrix / norms

    # Weighted normalised matrix
    weighted = norm_matrix * weights

    # Ideal and anti-ideal solutions
    ideal = np.where(benefit, weighted.max(axis=0), weighted.min(axis=0))
    anti_ideal = np.where(benefit, weighted.min(axis=0), weighted.max(axis=0))

    # Distances
    d_ideal = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_anti = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))

    # Closeness coefficient
    cc = d_anti / (d_ideal + d_anti + 1e-10)
    return cc


def run_mcdm_analysis(pred_df: pd.DataFrame) -> dict:
    """Run MCDM analysis with multiple weighting schemes."""
    logger.info("Running MCDM analysis")

    # Decision matrix: expected values for each target
    criteria_cols = [f"{t}_expected" for t in TARGET_COLS]
    if not all(c in pred_df.columns for c in criteria_cols):
        # Fallback to point predictions
        criteria_cols = TARGET_COLS
    dm = pred_df[criteria_cols].astype(float).values
    scenario_names = pred_df.index.tolist()
    criteria_labels = ["Recycling Rate", "Waste Reduction", "CO₂ Reduction"]

    # All criteria are benefit (higher = better)
    benefit = np.array([True, True, True])

    # Weighting schemes
    weight_schemes = {}

    # Equal weights
    weight_schemes["Equal"] = np.array([1/3, 1/3, 1/3])

    # Entropy method
    try:
        # Normalise columns to [0, 1]
        dm_norm = (dm - dm.min(axis=0)) / (dm.max(axis=0) - dm.min(axis=0) + 1e-10)
        # Add small epsilon to avoid log(0)
        dm_norm = np.clip(dm_norm, 1e-10, None)
        # Compute entropy
        p = dm_norm / dm_norm.sum(axis=0)
        k = 1 / np.log(dm.shape[0])
        entropy = -k * (p * np.log(p)).sum(axis=0)
        # Divergence
        div = 1 - entropy
        # Weights
        entropy_weights = div / div.sum()
        weight_schemes["Entropy"] = entropy_weights
        logger.info("  Entropy weights: %s", entropy_weights)
    except Exception as e:
        logger.warning("  Entropy weighting failed: %s", e)
        weight_schemes["Entropy"] = weight_schemes["Equal"]

    # Expert AHP weights (hypothetical — emphasising CO2 reduction)
    weight_schemes["AHP (CO₂ focus)"] = np.array([0.20, 0.30, 0.50])
    weight_schemes["AHP (Circularity focus)"] = np.array([0.50, 0.30, 0.20])

    # Run TOPSIS for each weighting scheme
    rankings = {}
    for scheme_name, weights in weight_schemes.items():
        cc = topsis(dm, weights, benefit)
        rank = np.argsort(-cc) + 1  # 1-based ranking
        rankings[scheme_name] = {
            "closeness_coeff": cc,
            "rank": rank,
        }
        logger.info("  %s weights: %s", scheme_name, dict(zip(scenario_names, rank)))

    # Build ranking table
    rank_data = {"Scenario": scenario_names}
    for scheme_name, r in rankings.items():
        rank_data[f"CC ({scheme_name})"] = r["closeness_coeff"]
        rank_data[f"Rank ({scheme_name})"] = r["rank"]
    rank_df = pd.DataFrame(rank_data)
    rank_df.to_csv(TABLES_DIR / "mcdm_ranking.csv", index=False, float_format="%.4f")
    logger.info("MCDM rankings:\n%s", rank_df.to_string())

    # Sensitivity analysis: vary weights ±20%
    sensitivity_results = _sensitivity_analysis(dm, weight_schemes, benefit, scenario_names)

    return {
        "ranking_df": rank_df,
        "weight_schemes": weight_schemes,
        "sensitivity": sensitivity_results,
    }


def _sensitivity_analysis(
    dm: np.ndarray,
    weight_schemes: dict,
    benefit: np.ndarray,
    scenario_names: list[str],
) -> pd.DataFrame:
    """Sensitivity analysis: vary each criterion weight ±20%."""
    logger.info("Running sensitivity analysis (±20% weight variation)")
    base_weights = weight_schemes["Equal"]
    n_criteria = len(base_weights)

    results = []
    for crit_idx in range(n_criteria):
        for delta in [-0.20, -0.10, 0, 0.10, 0.20]:
            weights = base_weights.copy()
            weights[crit_idx] *= (1 + delta)
            # Re-normalise
            weights = weights / weights.sum()
            cc = topsis(dm, weights, benefit)
            rank = np.argsort(-cc) + 1
            for s_idx, s_name in enumerate(scenario_names):
                results.append({
                    "criterion_varied": crit_idx,
                    "delta": delta,
                    "scenario": s_name,
                    "closeness_coeff": cc[s_idx],
                    "rank": rank[s_idx],
                })

    sens_df = pd.DataFrame(results)
    sens_df.to_csv(TABLES_DIR / "mcdm_sensitivity.csv", index=False, float_format="%.4f")

    # Sensitivity heatmap
    _plot_sensitivity_heatmap(sens_df, scenario_names)

    return sens_df


def _plot_sensitivity_heatmap(sens_df: pd.DataFrame, scenario_names: list[str]) -> None:
    """Plot sensitivity analysis heatmap."""
    criteria_names = ["Recycling Rate", "Waste Reduction", "CO₂ Reduction"]

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE, sharey=True)
    for crit_idx, (ax, crit_name) in enumerate(zip(axes, criteria_names)):
        crit_data = sens_df[sens_df["criterion_varied"] == crit_idx]
        pivot = crit_data.pivot_table(
            values="rank", index="scenario", columns="delta",
        )
        # Sort scenarios by name for consistent display
        pivot = pivot.reindex(scenario_names)
        sns.heatmap(
            pivot, annot=True, fmt=".0f", cmap="YlOrRd_r",
            ax=ax, cbar=crit_idx == 2,
            xticklabels=[f"{d:+.0%}" for d in sorted(crit_data["delta"].unique())],
        )
        ax.set_title(f"Vary: {crit_name}")
        ax.set_xlabel("Weight Change")
        if crit_idx == 0:
            ax.set_ylabel("Scenario")
        else:
            ax.set_ylabel("")

    fig.suptitle("Sensitivity Analysis: Rank Stability Under Weight Variation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mcdm_sensitivity_heatmap.pdf", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "mcdm_sensitivity_heatmap.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sensitivity heatmap")


def plot_scenario_comparison(scenario_df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    """Bar chart comparing scenario outcomes."""
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE, sharey=True)
    target_labels = ["Recycling Rate", "Waste Reduction", "CO₂ Reduction"]
    expected_cols = [f"{t}_expected" for t in TARGET_COLS]

    colors = sns.color_palette(COLOR_PALETTE, n_colors=len(scenario_df))
    short_names = [s.split(":")[0] for s in scenario_df.index]

    for ax, col, label in zip(axes, expected_cols, target_labels):
        if col in pred_df.columns:
            vals = pred_df[col].values
        else:
            vals = pred_df[TARGET_COLS[expected_cols.index(col)]].astype(float).values
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
        ax.set_title(label)
        ax.set_ylabel("Expected Score" if ax == axes[0] else "")
        ax.set_ylim(0, 5.5)

    fig.suptitle("Predicted Sustainability Outcomes by Scenario", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scenario_comparison.pdf", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "scenario_comparison.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scenario comparison plot")


def run_mcdm_pipeline(df: pd.DataFrame, feature_cols: list[str] | None = None) -> dict:
    """Run the full MCDM pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 5: MCDM SCENARIO ANALYSIS")
    logger.info("=" * 60)

    # Define scenarios
    scenarios = define_scenarios()
    logger.info("Defined %d scenarios", len(scenarios))

    # Predict outcomes
    pred_df = predict_scenario_outcomes(scenarios, df, feature_cols or [])

    # MCDM ranking
    mcdm_results = run_mcdm_analysis(pred_df)

    # Visualisation
    plot_scenario_comparison(scenarios, pred_df)

    logger.info("MCDM pipeline complete")
    return mcdm_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = pd.read_csv("dataset_clean.csv")
    run_mcdm_pipeline(df)
