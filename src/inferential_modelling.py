"""Phase 3: Inferential Modelling — Ordinal Logistic Regression and PLS-SEM."""
import logging
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    BIM_USE_COLS,
    COMPANY_SIZE_ORDER,
    FIGURE_DPI,
    FIGURES_DIR,
    FIGSIZE_SINGLE,
    PROJECT_SIZE_ORDER,
    TARGET_COLS,
    TABLES_DIR,
)

logger = logging.getLogger(__name__)


def _encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode ordinal/categorical features for regression."""
    data = df.copy()
    data["project_size_ord"] = data["project_size"].map(PROJECT_SIZE_ORDER)
    data["company_size_ord"] = data["company_size"].map(COMPANY_SIZE_ORDER)

    # Country dummies (reference: Croatia)
    country_dummies = pd.get_dummies(data["country"], prefix="country", drop_first=True, dtype=float)
    data = pd.concat([data, country_dummies], axis=1)
    return data


def run_ordinal_regression(df: pd.DataFrame) -> dict:
    """Run Ordinal Logistic Regression for each target variable."""
    logger.info("Running Ordinal Logistic Regression")

    from statsmodels.miscmodels.ordinal_model import OrderedModel

    data = _encode_ordinal_features(df)

    # Predictor columns
    predictors = BIM_USE_COLS + ["project_size_ord", "company_size_ord"]
    # Add country dummies if they exist
    country_cols = [c for c in data.columns if c.startswith("country_")]
    predictors += country_cols

    results = {}
    for target in TARGET_COLS:
        logger.info("  Target: %s", target)
        y = data[target].astype(float).values
        X = data[predictors].astype(float)

        # Drop rows with NaN
        mask = X.notna().all(axis=1) & pd.notna(y)
        X_clean = X[mask]
        y_clean = y[mask]

        if len(np.unique(y_clean)) < 2:
            logger.warning("  Skipping %s: fewer than 2 unique target values", target)
            continue

        try:
            # Remove constant or near-constant columns
            X_var = X_clean.loc[:, X_clean.std() > 1e-8]
            if X_var.shape[1] == 0:
                logger.warning("  No non-constant predictors for %s", target)
                continue
            # Iteratively ensure full rank by dropping linearly dependent columns
            from numpy.linalg import matrix_rank
            while matrix_rank(X_var.values) < X_var.shape[1] and X_var.shape[1] > 1:
                # Drop column with highest VIF / correlation
                corr_matrix = X_var.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                max_col = upper.max().idxmax()
                logger.info("    Dropping rank-deficient predictor: %s", max_col)
                X_var = X_var.drop(columns=[max_col])

            logger.info("    Using %d predictors (rank=%d)", X_var.shape[1], matrix_rank(X_var.values))

            model = OrderedModel(y_clean, X_var, distr="logit", hasconst=False)
            res = model.fit(method="bfgs", disp=False, maxiter=5000)

            # Extract results
            summary_df = pd.DataFrame({
                "Coefficient": res.params,
                "Std Error": res.bse,
                "z-value": res.tvalues,
                "p-value": res.pvalues,
                "OR": np.exp(res.params),
            })

            # Separate thresholds from predictors
            n_used_preds = X_var.shape[1]
            n_thresholds = len(res.params) - n_used_preds
            pred_results = summary_df.iloc[n_thresholds:]
            threshold_results = summary_df.iloc[:n_thresholds]

            # Pseudo R-squared
            pseudo_r2_mcfadden = 1 - (res.llf / res.llnull) if hasattr(res, "llnull") and res.llnull != 0 else np.nan

            result = {
                "model": res,
                "predictor_results": pred_results,
                "threshold_results": threshold_results,
                "pseudo_r2_mcfadden": pseudo_r2_mcfadden,
                "log_likelihood": res.llf,
                "aic": res.aic if hasattr(res, "aic") else np.nan,
                "bic": res.bic if hasattr(res, "bic") else np.nan,
                "n_obs": len(y_clean),
                "converged": res.mle_retvals.get("converged", True) if hasattr(res, "mle_retvals") else True,
            }
            results[target] = result

            logger.info("    McFadden R² = %.4f, AIC = %.2f, n = %d",
                        pseudo_r2_mcfadden, result["aic"], len(y_clean))
            logger.info("    Significant predictors (p < 0.05):")
            sig = pred_results[pred_results["p-value"] < 0.05]
            for idx, row in sig.iterrows():
                logger.info("      %s: β=%.3f, OR=%.3f, p=%.4f", idx, row["Coefficient"], row["OR"], row["p-value"])

        except Exception as e:
            logger.warning("  Ordinal regression failed for %s: %s", target, e)
            results[target] = {"error": str(e)}

    # Save consolidated results
    all_pred_results = []
    for target, res in results.items():
        if "predictor_results" in res:
            tdf = res["predictor_results"].copy()
            tdf["target"] = target
            tdf["predictor"] = tdf.index
            all_pred_results.append(tdf)
    if all_pred_results:
        combined = pd.concat(all_pred_results, ignore_index=True)
        combined.to_csv(TABLES_DIR / "ordinal_regression_results.csv", index=False, float_format="%.4f")

    # Model fit summary
    fit_summary = []
    for target, res in results.items():
        if "error" not in res:
            fit_summary.append({
                "Target": target,
                "N": res["n_obs"],
                "McFadden R²": res["pseudo_r2_mcfadden"],
                "AIC": res["aic"],
                "BIC": res["bic"],
                "Log-Likelihood": res["log_likelihood"],
                "Converged": res["converged"],
            })
    if fit_summary:
        pd.DataFrame(fit_summary).to_csv(TABLES_DIR / "ordinal_regression_fit.csv", index=False, float_format="%.4f")

    return results


def run_pls_sem(df: pd.DataFrame) -> dict:
    """Run PLS-SEM path model using semopy."""
    logger.info("Running PLS-SEM path model")

    try:
        import semopy

        data = _encode_ordinal_features(df)
        model_cols = BIM_USE_COLS + TARGET_COLS + ["project_size_ord", "company_size_ord"]
        model_data = data[model_cols].astype(float).dropna()

        # Define SEM model
        # Measurement model: BIM maturity -> 6 indicators, Sustainability -> 3 indicators
        # Structural model: BIM maturity -> Sustainability, controls -> Sustainability
        model_desc = """
        # Measurement model
        BIM_maturity =~ bim_feasibility_study + bim_energy_sustainability + bim_construction_mgmt + bim_space_tracking + bim_demolition + bim_recycling
        Sust_impact =~ sust_recycling_rate + sust_waste_reduction + sust_co2_reduction

        # Structural model
        Sust_impact ~ BIM_maturity + project_size_ord + company_size_ord
        """

        model = semopy.Model(model_desc)
        result = model.fit(model_data)

        # Get estimates
        estimates = model.inspect()
        estimates.to_csv(TABLES_DIR / "pls_sem_estimates.csv", index=False, float_format="%.4f")

        # Fit statistics
        try:
            fit_stats = semopy.calc_stats(model)
            fit_stats.to_csv(TABLES_DIR / "pls_sem_fit_stats.csv", float_format="%.4f")
            logger.info("PLS-SEM fit statistics:\n%s", fit_stats.to_string())
        except Exception as e:
            logger.warning("Could not compute SEM fit statistics: %s", e)
            fit_stats = None

        # Extract path coefficients
        structural = estimates[estimates["op"] == "~"]
        logger.info("PLS-SEM path coefficients:\n%s", structural.to_string())

        # Generate path diagram
        _plot_path_diagram(estimates)

        return {
            "estimates": estimates,
            "fit_stats": fit_stats,
            "structural_paths": structural,
            "converged": True,
        }

    except Exception as e:
        logger.warning("PLS-SEM failed: %s", e)
        return {"converged": False, "error": str(e)}


def _plot_path_diagram(estimates: pd.DataFrame) -> None:
    """Generate a simple path diagram visualisation."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Draw constructs
    from matplotlib.patches import FancyBboxPatch

    # BIM Maturity box
    bim_box = FancyBboxPatch((0.5, 4), 3, 2, boxstyle="round,pad=0.2",
                               facecolor="#E8F0FE", edgecolor="#1A73E8", linewidth=2)
    ax.add_patch(bim_box)
    ax.text(2, 5, "BIM\nMaturity", ha="center", va="center", fontsize=12, fontweight="bold")

    # Sustainability Impact box
    sust_box = FancyBboxPatch((6.5, 4), 3, 2, boxstyle="round,pad=0.2",
                                facecolor="#E6F4EA", edgecolor="#34A853", linewidth=2)
    ax.add_patch(sust_box)
    ax.text(8, 5, "Sustainability\nImpact", ha="center", va="center", fontsize=12, fontweight="bold")

    # Arrow from BIM to Sustainability
    structural = estimates[estimates["op"] == "~"]
    bim_path = structural[structural["rval"] == "BIM_maturity"]
    if len(bim_path) > 0:
        coef = bim_path["Estimate"].values[0]
        p_val = bim_path["p-value"].values[0] if "p-value" in bim_path.columns else np.nan
        ax.annotate("", xy=(6.5, 5), xytext=(3.5, 5),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#1A73E8"))
        sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        ax.text(5, 5.5, f"β={coef:.3f}{sig_marker}", ha="center", va="bottom", fontsize=11)

    # Controls
    for i, ctrl in enumerate(["project_size_ord", "company_size_ord"]):
        y_pos = 2 - i * 1.5
        ctrl_label = ctrl.replace("_ord", "").replace("_", " ").title()
        ax.text(5, y_pos, ctrl_label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor="#F9A825"))
        ctrl_path = structural[structural["rval"] == ctrl]
        if len(ctrl_path) > 0:
            coef = ctrl_path["Estimate"].values[0]
            ax.annotate("", xy=(6.5, 4), xytext=(5.5, y_pos + 0.3),
                         arrowprops=dict(arrowstyle="->", lw=1.5, color="#F9A825", ls="--"))
            ax.text(6, y_pos + 0.8, f"β={coef:.3f}", ha="center", fontsize=9, color="#F9A825")

    # BIM indicators (left side)
    bim_labels = ["Feasibility", "Energy", "Constr.Mgmt", "Space", "Demolition", "Recycling"]
    for i, label in enumerate(bim_labels):
        y = 8 - i * 1.0
        ax.text(0.5, y, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#F0F0F0", edgecolor="gray"))
        ax.annotate("", xy=(0.5, 6), xytext=(0.5, y - 0.3),
                     arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"))

    # Sustainability indicators (right side)
    sust_labels = ["Recycling Rate", "Waste Reduction", "CO₂ Reduction"]
    for i, label in enumerate(sust_labels):
        y = 8 - i * 1.2
        ax.text(9.5, y, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#F0F0F0", edgecolor="gray"))
        ax.annotate("", xy=(9.5, 6), xytext=(9.5, y - 0.3),
                     arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"))

    ax.set_title("PLS-SEM Path Model: BIM Maturity → Sustainability Impact", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pls_sem_path_diagram.pdf", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "pls_sem_path_diagram.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PLS-SEM path diagram")


def run_inferential_modelling(df: pd.DataFrame) -> dict:
    """Run the full inferential modelling pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 3: INFERENTIAL MODELLING")
    logger.info("=" * 60)

    results = {}

    # Ordinal Logistic Regression
    olr_results = run_ordinal_regression(df)
    results["ordinal_regression"] = olr_results

    # PLS-SEM
    sem_results = run_pls_sem(df)
    results["pls_sem"] = sem_results

    logger.info("Inferential modelling complete")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    warnings.filterwarnings("ignore")
    df = pd.read_csv("dataset_clean.csv")
    run_inferential_modelling(df)
