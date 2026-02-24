"""Main orchestrator: runs all 6 phases of the BIM-LCA-Circularity project."""
import logging
import sys
import time
import warnings

import numpy as np

from src.config import RANDOM_SEED, RESULTS_DIR

# Set seeds for reproducibility
np.random.seed(RANDOM_SEED)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / "pipeline.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def main() -> None:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("BIM-LCA-CIRCULARITY AI PREDICTIVE MODEL — FULL PIPELINE")
    logger.info("=" * 70)

    # ── Phase 1a: Data Loading & Cleaning ──────────────────────────────
    logger.info("PHASE 1a: DATA LOADING & CLEANING")
    from src.data_loading import run_data_pipeline
    df = run_data_pipeline()
    logger.info("Clean dataset: %d rows × %d columns", *df.shape)

    # ── Phase 1b: Exploratory Data Analysis ────────────────────────────
    logger.info("PHASE 1b: EXPLORATORY DATA ANALYSIS")
    from src.eda import run_eda
    run_eda(df)

    # ── Phase 2: Construct Validation ──────────────────────────────────
    logger.info("PHASE 2: CONSTRUCT VALIDATION")
    from src.construct_validation import run_construct_validation
    cv_results = run_construct_validation(df)

    # Check reliability threshold
    rel_df = cv_results.get("reliability")
    if rel_df is not None:
        for _, row in rel_df.iterrows():
            alpha = row.get("Cronbach α", 0)
            if isinstance(alpha, (int, float)) and alpha < 0.60:
                logger.warning(
                    "LOW RELIABILITY: %s (α=%.3f). Proceeding with caution.",
                    row.get("Construct", "?"), alpha,
                )

    # ── Phase 3: Inferential Modelling ─────────────────────────────────
    logger.info("PHASE 3: INFERENTIAL MODELLING")
    from src.inferential_modelling import run_inferential_modelling
    inf_results = run_inferential_modelling(df)

    # ── Phase 4: Predictive AI Modelling ───────────────────────────────
    logger.info("PHASE 4: PREDICTIVE AI MODELLING")
    from src.predictive_modelling import run_predictive_modelling
    pred_results = run_predictive_modelling(df)

    # ── Phase 5: MCDM Scenario Analysis ───────────────────────────────
    logger.info("PHASE 5: MCDM SCENARIO ANALYSIS")
    from src.mcdm import run_mcdm_pipeline
    mcdm_results = run_mcdm_pipeline(df, pred_results.get("feature_cols"))

    # ── Phase 6: Validation & Self-Challenge ───────────────────────────
    logger.info("PHASE 6: VALIDATION & PUBLICATION READINESS")
    from src.validation import run_validation
    model_results_df = pred_results.get("results_df")
    val_results = run_validation(df, model_results_df)

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE in %.1f seconds", elapsed)
    logger.info("Results saved to: %s", RESULTS_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
