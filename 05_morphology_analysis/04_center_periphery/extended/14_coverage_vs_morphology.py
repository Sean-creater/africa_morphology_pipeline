"""
14_coverage_vs_morphology.py
=============================
Q4 (embedding framework, panel b) — does comprehensive urban form carry
information about where people live beyond simple built-up coverage?

Two models predicting grid-cell log population density:
  M1 (baseline): log_dens ~ LCF_Built + country FE
  M2 (full)    : log_dens ~ LCF_Built + embedding[256] + country FE

log_dens uses Pop2022 directly (NOT divided by an assumed hex area): all
cells are the same 250-m hex size, so that area is a constant multiplier
that cancels exactly in both the R^2 comparison and the percentile-ratio
statistic below — log(Pop2022) and log(density) differ only by that
constant, so using Pop2022 avoids a fabricated-precision area constant
without losing anything the two statistics below actually need.

Both models are Ridge-regularised (256 correlated embedding dims is not
something plain OLS handles well) and evaluated with GROUPED (by city)
K-fold cross-validation — a random row split would leak information
between spatially autocorrelated cells in the same city, inflating
out-of-sample R^2 for both models equally but especially for M2, which
has far more capacity to overfit city-specific patterns.

Headline statistics
--------------------
  delta_r2        = oos_R2(M2) - oos_R2(M1)   [out-of-sample, averaged
                     across folds]
  density_ratio_90_10
      At matched LCF_Built, how much does comprehensive morphology alone
      shift predicted density? Computed from M2's fitted coefficients:
      embedding_score_i = sum_j beta_j * embedding_ij (the embedding
      terms' contribution only, holding LCF_Built/country fixed — this
      is exactly the part of M2's prediction NOT explained by coverage).
      density_ratio_90_10 = exp(P90(embedding_score) - P10(embedding_score)),
      i.e. the population-density fold-difference between a cell at the
      90th vs 10th percentile of morphology-driven variation, both at the
      same coverage level.

Output
------
  outputs/tables/coverage_vs_morphology.csv
      model | fold | r2_oos
  outputs/tables/coverage_vs_morphology_summary.csv
      statistic | value

Usage
-----
    python 14_coverage_vs_morphology.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).parent
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_ARRAYS = ROOT / "outputs" / "arrays"
INDEX_PQ   = OUT_TABLES / "cell_embeddings_index.parquet"
EMB_NPY    = OUT_ARRAYS / "cell_embeddings.npy"

N_FOLDS       = 5
RIDGE_ALPHAS  = np.logspace(-2, 4, 25)
RNG_SEED      = 42


def _country_fe_matrix(country: pd.Series) -> np.ndarray:
    """One-hot country dummies, reference = most frequent country (dropped)."""
    dummies = pd.get_dummies(country, drop_first=True)
    return dummies.values.astype(np.float32), dummies.columns.tolist()


def main():
    log.info("Loading cell_embeddings_index.parquet + cell_embeddings.npy ...")
    idx = pd.read_parquet(INDEX_PQ,
                          columns=["GridID", "city", "country", "row_idx",
                                   "Pop2022", "LCF_Built"])
    emb = np.load(EMB_NPY)
    log.info("  %d cells, embedding dim=%d.", len(idx), emb.shape[1])

    # Drop rows with missing coverage or non-positive population (log undefined)
    mask = idx["LCF_Built"].notna() & (idx["Pop2022"] > 0)
    n_drop = (~mask).sum()
    log.info("  Dropping %d cells (missing LCF_Built or Pop2022<=0).", n_drop)
    idx = idx[mask].reset_index(drop=True)
    emb = emb[idx["row_idx"].values]

    y = np.log(idx["Pop2022"].values)
    coverage = idx[["LCF_Built"]].values.astype(np.float32)
    country_fe, country_cols = _country_fe_matrix(idx["country"])
    groups = idx["city"].astype(str) + "_" + idx["country"].astype(str)

    X_m1 = np.hstack([coverage, country_fe])
    X_m2 = np.hstack([coverage, emb, country_fe])

    log.info("Running %d-fold grouped (by city) CV for M1 (coverage only) "
            "and M2 (coverage + embedding) ...", N_FOLDS)
    gkf = GroupKFold(n_splits=N_FOLDS)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_m1, y, groups)):
        for model_name, X in [("M1_coverage", X_m1), ("M2_coverage_embedding", X_m2)]:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=3)
            ridge.fit(X_tr, y[train_idx])
            r2 = ridge.score(X_te, y[test_idx])
            rows.append({"model": model_name, "fold": fold, "r2_oos": r2,
                        "alpha": ridge.alpha_})
            log.info("  fold %d  %-22s  R2_oos=%.4f  (alpha=%.3g)",
                     fold, model_name, r2, ridge.alpha_)

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(OUT_TABLES / "coverage_vs_morphology.csv",
                   index=False, float_format="%.6f")
    log.info("Saved: coverage_vs_morphology.csv")

    r2_m1 = fold_df.loc[fold_df["model"] == "M1_coverage", "r2_oos"].mean()
    r2_m2 = fold_df.loc[fold_df["model"] == "M2_coverage_embedding", "r2_oos"].mean()
    delta_r2 = r2_m2 - r2_m1
    pct_increase = 100 * delta_r2 / r2_m1 if r2_m1 > 0 else np.nan

    # ── Density ratio at matched coverage: refit M2 on ALL data (the
    # percentile-ratio statistic describes the fitted relationship itself,
    # not out-of-sample generalisation, so no CV split needed here) ──
    log.info("Refitting M2 on full data for the embedding-contribution score ...")
    scaler = StandardScaler()
    X_m2_scaled = scaler.fit_transform(X_m2)
    ridge_full = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
    ridge_full.fit(X_m2_scaled, y)

    n_emb = emb.shape[1]
    emb_start, emb_end = 1, 1 + n_emb   # column 0 = coverage, then embedding, then country FE
    emb_coefs  = ridge_full.coef_[emb_start:emb_end]
    emb_scaled = X_m2_scaled[:, emb_start:emb_end]
    embedding_score = emb_scaled @ emb_coefs

    p10, p90 = np.percentile(embedding_score, [10, 90])
    density_ratio_90_10 = float(np.exp(p90 - p10))

    summary = pd.DataFrame([
        {"statistic": "r2_oos_M1_coverage",          "value": r2_m1},
        {"statistic": "r2_oos_M2_coverage_embedding", "value": r2_m2},
        {"statistic": "delta_r2",                     "value": delta_r2},
        {"statistic": "pct_increase_over_M1",         "value": pct_increase},
        {"statistic": "density_ratio_p90_p10_at_matched_coverage",
         "value": density_ratio_90_10},
        {"statistic": "ridge_alpha_full_fit",         "value": ridge_full.alpha_},
        {"statistic": "n_cells",                      "value": len(idx)},
        {"statistic": "n_cities",                     "value": groups.nunique()},
    ])
    summary.to_csv(OUT_TABLES / "coverage_vs_morphology_summary.csv",
                   index=False, float_format="%.6f")
    log.info("Saved: coverage_vs_morphology_summary.csv")

    print("\n=== Coverage vs. comprehensive morphology ===")
    print(summary.to_string(index=False))
    print(f"\nHeadline: at matched built-up coverage, comprehensive urban form "
         f"corresponds to a {density_ratio_90_10:.2f}x difference in population "
         f"density (90th vs 10th percentile of morphology-driven variation), "
         f"and raises out-of-sample R2 by {pct_increase:.1f}% over coverage alone "
         f"({r2_m1:.3f} -> {r2_m2:.3f}).")

    log.info("Done.")


if __name__ == "__main__":
    main()
