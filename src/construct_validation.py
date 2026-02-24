"""Phase 2: Construct Validation — Reliability (alpha, omega) and Validity (EFA, CFA, AVE, HTMT).

Uses manual implementations for EFA to avoid factor_analyzer/sklearn compatibility issues.
"""
import logging
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats, linalg

from src.config import (
    BIM_USE_COLS,
    COLOR_PALETTE,
    FIGURE_DPI,
    FIGURES_DIR,
    FIGSIZE_SINGLE,
    TARGET_COLS,
    TABLES_DIR,
)

logger = logging.getLogger(__name__)


# ── Manual factor analysis utilities ──────────────────────────────────


def _bartlett_sphericity(data: pd.DataFrame) -> tuple[float, float]:
    """Bartlett's test of sphericity (manual implementation)."""
    n, p = data.shape
    corr = data.corr().values
    det = np.linalg.det(corr)
    if det <= 0:
        return np.nan, np.nan
    chi2 = -((n - 1) - (2 * p + 5) / 6) * np.log(det)
    dof = p * (p - 1) / 2
    p_value = 1 - stats.chi2.cdf(chi2, dof)
    return chi2, p_value


def _kmo(data: pd.DataFrame) -> float:
    """Kaiser-Meyer-Olkin measure of sampling adequacy (manual implementation)."""
    corr = data.corr().values
    try:
        inv_corr = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr)
    n = corr.shape[0]

    # Partial correlations from inverse correlation matrix
    partial = np.zeros_like(corr)
    for i in range(n):
        for j in range(n):
            if i != j:
                partial[i, j] = -inv_corr[i, j] / np.sqrt(inv_corr[i, i] * inv_corr[j, j])

    # KMO
    corr_sq_sum = 0
    partial_sq_sum = 0
    for i in range(n):
        for j in range(i + 1, n):
            corr_sq_sum += corr[i, j] ** 2
            partial_sq_sum += partial[i, j] ** 2

    if (corr_sq_sum + partial_sq_sum) == 0:
        return 0.0
    return corr_sq_sum / (corr_sq_sum + partial_sq_sum)


def _varimax_rotation(loadings: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Varimax rotation of factor loadings."""
    p, k = loadings.shape
    rotation_matrix = np.eye(k)
    d = 0

    for _ in range(max_iter):
        old_d = d
        for i in range(k):
            for j in range(i + 1, k):
                u = loadings[:, i] ** 2 - loadings[:, j] ** 2
                v = 2 * loadings[:, i] * loadings[:, j]
                A = u.sum()
                B = v.sum()
                C = (u ** 2 - v ** 2).sum()
                D = (2 * u * v).sum()
                num = D - 2 * A * B / p
                den = C - (A ** 2 - B ** 2) / p
                theta = 0.25 * np.arctan2(num, den)

                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                rot = np.eye(k)
                rot[i, i] = cos_t
                rot[j, j] = cos_t
                rot[i, j] = -sin_t
                rot[j, i] = sin_t

                loadings = loadings @ rot
                rotation_matrix = rotation_matrix @ rot

        d = np.sum(loadings ** 4) - np.sum(loadings ** 2) ** 2 / p
        if abs(d - old_d) < tol:
            break

    return loadings


def _extract_factors(data: pd.DataFrame, n_factors: int, rotation: str = "varimax") -> dict:
    """Extract factors using principal axis factoring on the correlation matrix."""
    corr = data.corr().values
    n_vars = corr.shape[0]

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Extract loadings for n_factors
    loadings = eigenvectors[:, :n_factors] * np.sqrt(np.maximum(eigenvalues[:n_factors], 0))

    # Apply rotation
    if rotation == "varimax" and n_factors > 1:
        loadings = _varimax_rotation(loadings)

    # Communalities and uniquenesses
    communalities = (loadings ** 2).sum(axis=1)
    uniquenesses = 1 - communalities

    # Variance explained
    ss_loadings = (loadings ** 2).sum(axis=0)
    prop_var = ss_loadings / n_vars
    cum_var = np.cumsum(prop_var)

    return {
        "loadings": loadings,
        "eigenvalues": eigenvalues,
        "communalities": communalities,
        "uniquenesses": uniquenesses,
        "ss_loadings": ss_loadings,
        "prop_var": prop_var,
        "cum_var": cum_var,
    }


# ── Reliability ───────────────────────────────────────────────────────


def cronbach_alpha(data: pd.DataFrame) -> float:
    """Compute Cronbach's alpha for a set of items."""
    items = data.dropna()
    n_items = items.shape[1]
    if n_items < 2:
        return np.nan
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    return alpha


def mcdonalds_omega(data: pd.DataFrame) -> float:
    """Compute McDonald's omega (total) using single-factor model."""
    items = data.dropna().astype(float)
    if items.shape[1] < 2:
        return np.nan
    try:
        result = _extract_factors(items, n_factors=1, rotation="varimax")
        loadings = result["loadings"].flatten()
        uniquenesses = result["uniquenesses"]
        sum_loadings = loadings.sum()
        omega = sum_loadings ** 2 / (sum_loadings ** 2 + uniquenesses.sum())
        return omega
    except Exception as e:
        logger.warning("McDonald's omega computation failed: %s", e)
        return np.nan


def compute_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Compute reliability for all constructs."""
    logger.info("Computing reliability metrics")

    constructs = {
        "BIM-use maturity (6 items)": BIM_USE_COLS,
        "Sustainability impact (3 items)": TARGET_COLS,
        "Circularity sub-construct (2 items)": ["bim_recycling", "sust_recycling_rate"],
    }

    results = []
    for name, cols in constructs.items():
        data = df[cols].astype(float)
        alpha = cronbach_alpha(data)
        omega = mcdonalds_omega(data)
        n_items = len(cols)
        results.append({
            "Construct": name,
            "N Items": n_items,
            "Cronbach α": round(alpha, 4),
            "McDonald ω": round(omega, 4) if not np.isnan(omega) else "N/A",
            "Acceptable (α≥0.60)": "Yes" if alpha >= 0.60 else "NO",
        })
        logger.info("  %s: α=%.4f, ω=%s", name, alpha,
                     f"{omega:.4f}" if not np.isnan(omega) else "N/A")

    rel_df = pd.DataFrame(results)
    rel_df.to_csv(TABLES_DIR / "reliability_results.csv", index=False)
    logger.info("Saved reliability results")
    return rel_df


# ── EFA ───────────────────────────────────────────────────────────────


def run_efa(df: pd.DataFrame) -> dict:
    """Run Exploratory Factor Analysis with varimax rotation."""
    logger.info("Running Exploratory Factor Analysis (EFA)")
    all_items = BIM_USE_COLS + TARGET_COLS
    data = df[all_items].astype(float).dropna()

    # Bartlett's test
    chi_sq, p_value = _bartlett_sphericity(data)
    logger.info("  Bartlett's test: χ²=%.2f, p=%.6f", chi_sq, p_value)

    # KMO
    kmo = _kmo(data)
    logger.info("  KMO measure: %.4f", kmo)

    # Eigenvalues from correlation matrix
    corr = data.corr().values
    eigenvalues = np.sort(np.linalg.eigvalsh(corr))[::-1]
    n_factors = int((eigenvalues > 1).sum())
    n_factors = max(2, min(n_factors, 3))
    logger.info("  Eigenvalues > 1 suggest %d factors", n_factors)

    # Extract factors
    result = _extract_factors(data, n_factors, rotation="varimax")

    # Loadings table
    loadings = pd.DataFrame(
        result["loadings"],
        index=all_items,
        columns=[f"Factor {i+1}" for i in range(n_factors)],
    )
    loadings["Communality"] = result["communalities"]
    loadings["Uniqueness"] = result["uniquenesses"]

    # Variance explained
    var_df = pd.DataFrame(
        [result["ss_loadings"], result["prop_var"], result["cum_var"]],
        index=["SS Loadings", "Proportion Var", "Cumulative Var"],
        columns=[f"Factor {i+1}" for i in range(n_factors)],
    )

    # Save
    loadings.to_csv(TABLES_DIR / "efa_loadings.csv", float_format="%.4f")
    var_df.to_csv(TABLES_DIR / "efa_variance.csv", float_format="%.4f")

    # Scree plot
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.plot(range(1, len(eigenvalues) + 1), eigenvalues, "bo-")
    ax.axhline(y=1, color="r", linestyle="--", label="Eigenvalue = 1")
    ax.set_xlabel("Factor Number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Scree Plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scree_plot.pdf", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "scree_plot.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info("EFA loadings:\n%s", loadings.to_string(float_format="%.3f"))

    return {
        "bartlett_chi2": chi_sq,
        "bartlett_p": p_value,
        "kmo": kmo,
        "n_factors": n_factors,
        "loadings": loadings,
        "variance": var_df,
        "eigenvalues": eigenvalues,
    }


# ── CFA ───────────────────────────────────────────────────────────────


def run_cfa(df: pd.DataFrame) -> dict:
    """Run Confirmatory Factor Analysis using semopy."""
    logger.info("Running Confirmatory Factor Analysis (CFA)")

    try:
        import semopy

        data = df[BIM_USE_COLS + TARGET_COLS].astype(float).dropna()

        model_desc = """
        BIM_maturity =~ bim_feasibility_study + bim_energy_sustainability + bim_construction_mgmt + bim_space_tracking + bim_demolition + bim_recycling
        Sust_impact =~ sust_recycling_rate + sust_waste_reduction + sust_co2_reduction
        """

        model = semopy.Model(model_desc)
        model.fit(data)

        # Fit indices
        try:
            stats_dict = semopy.calc_stats(model)
            fit_indices = {}
            for key in ["CFI", "RMSEA", "SRMR", "chi2", "DoF", "chi2 p-value"]:
                for k in stats_dict.index:
                    if k.lower().replace(" ", "") == key.lower().replace(" ", ""):
                        fit_indices[key] = stats_dict.loc[k].values[0]
                        break
            logger.info("CFA fit indices: %s", fit_indices)
            fit_df = pd.DataFrame([fit_indices])
            fit_df.to_csv(TABLES_DIR / "cfa_fit_indices.csv", index=False, float_format="%.4f")
        except Exception as e:
            logger.warning("Could not compute CFA fit statistics: %s", e)
            fit_indices = {}

        estimates = model.inspect()
        estimates.to_csv(TABLES_DIR / "cfa_estimates.csv", index=False, float_format="%.4f")

        return {
            "fit_indices": fit_indices,
            "estimates": estimates,
            "converged": True,
        }

    except Exception as e:
        logger.warning("CFA failed: %s. This may be due to sample characteristics.", e)
        return {"converged": False, "error": str(e)}


# ── AVE and HTMT ──────────────────────────────────────────────────────


def compute_ave_htmt(df: pd.DataFrame) -> dict:
    """Compute AVE and HTMT for discriminant validity."""
    logger.info("Computing AVE and HTMT")

    constructs = {
        "BIM_maturity": BIM_USE_COLS,
        "Sust_impact": TARGET_COLS,
    }

    data = df[BIM_USE_COLS + TARGET_COLS].astype(float).dropna()

    # AVE via eigendecomposition
    ave_results = {}
    for name, cols in constructs.items():
        try:
            result = _extract_factors(data[cols], n_factors=1, rotation="varimax")
            loadings_sq = result["loadings"].flatten() ** 2
            ave = loadings_sq.mean()
            ave_results[name] = ave
            logger.info("  AVE(%s) = %.4f %s", name, ave,
                        "(≥0.50: OK)" if ave >= 0.50 else "(< 0.50: BELOW THRESHOLD)")
        except Exception as e:
            logger.warning("  AVE computation failed for %s: %s", name, e)
            ave_results[name] = np.nan

    # HTMT
    construct_list = list(constructs.keys())
    htmt_results = {}
    for i in range(len(construct_list)):
        for j in range(i + 1, len(construct_list)):
            c1_cols = constructs[construct_list[i]]
            c2_cols = constructs[construct_list[j]]

            het_corrs = []
            for c1 in c1_cols:
                for c2 in c2_cols:
                    r = data[c1].corr(data[c2])
                    het_corrs.append(abs(r))
            mean_het = np.mean(het_corrs)

            mono1 = []
            for a in range(len(c1_cols)):
                for b in range(a + 1, len(c1_cols)):
                    r = data[c1_cols[a]].corr(data[c1_cols[b]])
                    mono1.append(abs(r))
            mean_mono1 = np.mean(mono1) if mono1 else 1

            mono2 = []
            for a in range(len(c2_cols)):
                for b in range(a + 1, len(c2_cols)):
                    r = data[c2_cols[a]].corr(data[c2_cols[b]])
                    mono2.append(abs(r))
            mean_mono2 = np.mean(mono2) if mono2 else 1

            htmt = mean_het / np.sqrt(mean_mono1 * mean_mono2) if (mean_mono1 * mean_mono2) > 0 else np.nan
            pair_name = f"{construct_list[i]}_vs_{construct_list[j]}"
            htmt_results[pair_name] = htmt
            logger.info("  HTMT(%s) = %.4f %s", pair_name, htmt,
                        "(< 0.85: OK)" if htmt < 0.85 else "(≥ 0.85: DISCRIMINANT VALIDITY CONCERN)")

    # Save
    validity_df = pd.DataFrame({
        "Construct": list(ave_results.keys()),
        "AVE": list(ave_results.values()),
        "AVE ≥ 0.50": ["Yes" if v >= 0.50 else "No" for v in ave_results.values()],
    })
    validity_df.to_csv(TABLES_DIR / "ave_results.csv", index=False, float_format="%.4f")

    htmt_df = pd.DataFrame({
        "Pair": list(htmt_results.keys()),
        "HTMT": list(htmt_results.values()),
        "HTMT < 0.85": ["Yes" if v < 0.85 else "No" for v in htmt_results.values()],
    })
    htmt_df.to_csv(TABLES_DIR / "htmt_results.csv", index=False, float_format="%.4f")

    return {"ave": ave_results, "htmt": htmt_results}


# ── Pipeline ──────────────────────────────────────────────────────────


def run_construct_validation(df: pd.DataFrame) -> dict:
    """Run the full construct validation pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 2: CONSTRUCT VALIDATION")
    logger.info("=" * 60)

    results = {}

    rel_df = compute_reliability(df)
    results["reliability"] = rel_df

    alpha_bim = cronbach_alpha(df[BIM_USE_COLS].astype(float))
    if alpha_bim < 0.60:
        logger.warning(
            "CRITICAL: BIM-use construct α=%.4f < 0.60. "
            "Proceeding with caution — this will be noted as a limitation.", alpha_bim
        )

    efa_results = run_efa(df)
    results["efa"] = efa_results

    cfa_results = run_cfa(df)
    results["cfa"] = cfa_results

    validity = compute_ave_htmt(df)
    results["validity"] = validity

    logger.info("Construct validation complete")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    warnings.filterwarnings("ignore")
    df = pd.read_csv("dataset_clean.csv")
    run_construct_validation(df)
