"""Data loading, cleaning, and quality reporting for the BIM-LCA-Circularity project."""
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    BIM_USE_COLS,
    COLUMN_NAMES,
    DATA_CLEAN,
    DATA_RAW,
    TARGET_COLS,
    TABLES_DIR,
)

logger = logging.getLogger(__name__)


def _parse_countries_operated(val: Any) -> int | None:
    """Normalise column 2 (countries operated in) to integer."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip().lower()
    mapping = {
        "more than 5": 6,
        "in 1 country": 1,
        "in 2 countries": 2,
        "in 4 countries": 4,
    }
    for key, v in mapping.items():
        if key in s:
            return v
    # Try to extract a number
    import re
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    logger.warning("Could not parse countries_operated_in value: %r", val)
    return np.nan


def _simplify_project_size(val: str | None) -> str | None:
    """Map verbose project size labels to short form."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).lower()
    if "small" in s:
        return "Small"
    if "medium" in s:
        return "Medium"
    if "large" in s:
        return "Large"
    return val


def _simplify_company_size(val: str | None) -> str | None:
    """Map verbose company size labels to short form."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).lower()
    if "small" in s and "medium" not in s:
        return "Small"
    if "medium" in s:
        return "Medium"
    if "large" in s:
        return "Large"
    return val


def _simplify_activity(val: str | None) -> str | None:
    """Extract NACE code prefix from main construction activity."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip()
    if s.startswith("41.20.1"):
        return "41.20.1_Residential"
    if s.startswith("41.20.2"):
        return "41.20.2_NonResidential"
    if s.startswith("42.11"):
        return "42.11_Roads"
    if s.startswith("42.21"):
        return "42.21_Pipelines"
    if "residential" in s.lower():
        return "41.20.1_Residential"
    return s[:20]


def load_raw_data() -> pd.DataFrame:
    """Load raw Excel data, extract only valid data rows (skip headers and summary)."""
    logger.info("Loading raw data from %s", DATA_RAW)
    df = pd.read_excel(
        DATA_RAW,
        sheet_name=0,
        header=None,
        skiprows=3,  # Skip rows 1-3 (0-indexed: 0,1,2)
        nrows=201,   # Rows 4-204 in Excel = 201 rows
    )
    # Rename columns
    df.columns = [COLUMN_NAMES.get(i + 1, f"col_{i+1}") for i in range(df.shape[1])]
    logger.info("Loaded raw data: shape=%s", df.shape)
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the dataset and return (clean_df, quality_report)."""
    report: dict[str, Any] = {}
    n_initial = len(df)
    report["initial_rows"] = n_initial

    # 1. Drop column 10 (company_name) — mostly null
    df = df.drop(columns=["company_name"], errors="ignore")
    logger.info("Dropped company_name column")

    # 2. Detect and remove aggregate/mean rows
    # These have float (non-integer) values in Likert columns
    likert_cols = BIM_USE_COLS + TARGET_COLS
    mask_aggregate = pd.Series(False, index=df.index)
    for col in likert_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            is_float = vals.notna() & (vals != vals.round(0))
            mask_aggregate = mask_aggregate | is_float
    aggregate_rows = df[mask_aggregate].index.tolist()
    report["aggregate_rows_removed"] = len(aggregate_rows)
    logger.info("Removing %d aggregate/mean rows: %s", len(aggregate_rows), aggregate_rows)
    df = df[~mask_aggregate].copy()

    # 3. Remove fully null rows
    mask_null = df.isnull().all(axis=1)
    null_rows = df[mask_null].index.tolist()
    report["null_rows_removed"] = len(null_rows)
    logger.info("Removing %d fully null rows", len(null_rows))
    df = df[~mask_null].copy()

    # 4. Report duplicate profile count (kept as-is per user decision)
    n_unique = len(df.drop_duplicates())
    report["unique_profiles"] = n_unique
    report["duplicate_rows_kept"] = len(df) - n_unique
    logger.info(
        "Data contains %d unique profiles across %d rows (duplicates retained as valid observations)",
        n_unique, len(df),
    )

    # 5. Clean column 2: countries_operated_in
    df["countries_operated_in"] = df["countries_operated_in"].apply(_parse_countries_operated)

    # 6. Clean Likert columns: force to int, treat non-numeric as NaN
    for col in likert_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Check for values outside 1-5 range
        out_of_range = df[col].notna() & ((df[col] < 1) | (df[col] > 5))
        if out_of_range.any():
            logger.warning("Column %s has %d values outside 1-5 range", col, out_of_range.sum())
            df.loc[out_of_range, col] = np.nan
        # Convert to nullable integer
        df[col] = df[col].round(0).astype("Int64")

    # 7. Simplify categorical columns
    df["project_size"] = df["project_size"].apply(_simplify_project_size)
    df["company_size"] = df["company_size"].apply(_simplify_company_size)
    df["main_construction_activity"] = df["main_construction_activity"].apply(_simplify_activity)

    # 8. Final missing value report
    report["missing_per_column"] = df.isnull().sum().to_dict()
    report["final_shape"] = df.shape
    report["country_distribution"] = df["country"].value_counts().to_dict()

    # Reset index
    df = df.reset_index(drop=True)

    logger.info("Cleaning complete. Final shape: %s", df.shape)
    return df, report


def save_clean_data(df: pd.DataFrame) -> None:
    """Save cleaned dataset to CSV."""
    df.to_csv(DATA_CLEAN, index=False)
    logger.info("Saved clean dataset to %s", DATA_CLEAN)


def print_quality_report(report: dict) -> str:
    """Format and return a quality report string."""
    lines = [
        "=" * 60,
        "DATA QUALITY REPORT",
        "=" * 60,
        f"Initial rows loaded: {report['initial_rows']}",
        f"Aggregate/mean rows removed: {report['aggregate_rows_removed']}",
        f"Fully null rows removed: {report['null_rows_removed']}",
        f"Unique respondent profiles: {report['unique_profiles']}",
        f"Duplicate rows (retained): {report['duplicate_rows_kept']}",
        f"Final dataset shape: {report['final_shape']}",
        "",
        "Country distribution:",
    ]
    for c, n in sorted(report["country_distribution"].items()):
        lines.append(f"  {c}: {n}")
    lines.append("")
    lines.append("Missing values per column:")
    for col, n in report["missing_per_column"].items():
        if n > 0:
            lines.append(f"  {col}: {n}")
    if not any(v > 0 for v in report["missing_per_column"].values()):
        lines.append("  (none)")
    lines.append("=" * 60)
    return "\n".join(lines)


def run_data_pipeline() -> pd.DataFrame:
    """Execute the full data loading and cleaning pipeline."""
    raw = load_raw_data()
    clean, report = clean_data(raw)
    report_text = print_quality_report(report)
    logger.info("\n%s", report_text)

    # Save report
    report_path = TABLES_DIR / "data_quality_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    save_clean_data(clean)
    return clean


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = run_data_pipeline()
    print(df.head())
    print(df.dtypes)
