"""Exploratory Data Analysis: descriptive stats, visualisations, correlation analysis."""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    BIM_USE_COLS,
    CATEGORICAL_FEATURES,
    COLOR_PALETTE,
    FIGURE_DPI,
    FIGURES_DIR,
    FIGSIZE_LARGE,
    FIGSIZE_SINGLE,
    FIGSIZE_WIDE,
    TARGET_COLS,
    TABLES_DIR,
)

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette=COLOR_PALETTE)


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table 1: descriptive statistics for all variables."""
    logger.info("Computing descriptive statistics")

    # Numeric columns
    numeric_cols = BIM_USE_COLS + TARGET_COLS + ["countries_operated_in"]
    desc_num = df[numeric_cols].describe().T
    desc_num["median"] = df[numeric_cols].median()
    desc_num["skew"] = df[numeric_cols].skew()
    desc_num["kurtosis"] = df[numeric_cols].kurtosis()
    desc_num = desc_num[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "median", "skew", "kurtosis"]]

    # Save
    desc_num.to_csv(TABLES_DIR / "descriptive_statistics_numeric.csv", float_format="%.3f")
    logger.info("Saved numeric descriptive statistics")

    # Categorical columns
    cat_stats = []
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            vc = df[col].value_counts()
            for val, count in vc.items():
                cat_stats.append({
                    "variable": col,
                    "category": val,
                    "count": count,
                    "percentage": count / len(df) * 100,
                })
    desc_cat = pd.DataFrame(cat_stats)
    desc_cat.to_csv(TABLES_DIR / "descriptive_statistics_categorical.csv", index=False, float_format="%.1f")
    logger.info("Saved categorical descriptive statistics")

    return desc_num


def plot_target_distributions(df: pd.DataFrame) -> None:
    """Plot class distributions for the three target variables."""
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
    target_labels = {
        "sust_recycling_rate": "Recycling Rate Increase",
        "sust_waste_reduction": "Waste Reduction",
        "sust_co2_reduction": "CO₂ Emission Reduction",
    }
    for ax, col in zip(axes, TARGET_COLS):
        counts = df[col].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=sns.color_palette(COLOR_PALETTE, n_colors=5))
        ax.set_xlabel("Likert Score (1–5)")
        ax.set_ylabel("Count")
        ax.set_title(target_labels.get(col, col))
        ax.set_xticks([1, 2, 3, 4, 5])
    fig.suptitle("Target Variable Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_distributions.pdf", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "target_distributions.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved target distribution plots")


def plot_bim_profiles_by_country(df: pd.DataFrame) -> None:
    """Radar/bar chart of BIM-use profiles by country."""
    bim_labels = [
        "Feasibility\nStudy",
        "Energy\nSustainability",
        "Construction\nMgmt",
        "Space\nTracking",
        "Demolition",
        "Recycling",
    ]

    means = df.groupby("country")[BIM_USE_COLS].mean()

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    x = np.arange(len(BIM_USE_COLS))
    width = 0.25
    countries = means.index.tolist()
    colors = sns.color_palette(COLOR_PALETTE, n_colors=len(countries))

    for i, (country, color) in enumerate(zip(countries, colors)):
        offset = (i - len(countries) / 2 + 0.5) * width
        vals = means.loc[country].values
        ax.bar(x + offset, vals, width, label=country, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(bim_labels, fontsize=9)
    ax.set_ylabel("Mean BIM-Use Score (1–5)")
    ax.set_title("BIM-Use Profiles by Country")
    ax.legend()
    ax.set_ylim(0, 5.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bim_profiles_by_country.pdf", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "bim_profiles_by_country.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved BIM profiles by country")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Spearman correlation heatmap for all ordinal variables."""
    ordinal_cols = BIM_USE_COLS + TARGET_COLS
    corr = df[ordinal_cols].astype(float).corr(method="spearman")

    short_labels = [
        "BIM:Feasibility",
        "BIM:Energy",
        "BIM:Construction",
        "BIM:Space",
        "BIM:Demolition",
        "BIM:Recycling",
        "Sust:RecyclingRate",
        "Sust:WasteReduct",
        "Sust:CO₂Reduct",
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=short_labels,
        yticklabels=short_labels,
        ax=ax,
        square=True,
    )
    ax.set_title("Spearman Correlation: BIM-Use & Sustainability Indicators", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.pdf", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved correlation heatmap")


def plot_country_distribution(df: pd.DataFrame) -> None:
    """Bar chart of sample distribution by country."""
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = df["country"].value_counts().sort_index()
    colors = sns.color_palette(COLOR_PALETTE, n_colors=len(vc))
    ax.bar(vc.index, vc.values, color=colors)
    ax.set_ylabel("Count")
    ax.set_title("Sample Distribution by Country")
    for i, (idx, val) in enumerate(vc.items()):
        ax.text(i, val + 1, str(val), ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "country_distribution.pdf", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "country_distribution.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved country distribution plot")


def run_eda(df: pd.DataFrame) -> None:
    """Run the full EDA pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 1b: EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    desc = descriptive_statistics(df)
    logger.info("Descriptive statistics (numeric):\n%s", desc.to_string())

    plot_target_distributions(df)
    plot_bim_profiles_by_country(df)
    plot_correlation_heatmap(df)
    plot_country_distribution(df)

    logger.info("EDA complete. All figures saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = pd.read_csv("dataset_clean.csv")
    run_eda(df)
