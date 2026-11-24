"""
00_define_size_tiers.py
=======================
Q4: Internal spatial organisation of urban morphology across city-size regimes.

Define four population-size tiers inherited from the Part 3 urban
hierarchy analysis, which identified 20k / 200k / 1m as breakpoints in
the population–morphology scaling curves (H17, r17 vs log Pop).

Tier thresholds (fixed, from Part 3)
--------------------------------------
  T1  Pop < 20 000          (ln P < 9.90)   small settlements
  T2  20 000 ≤ Pop < 200 000 (9.90–12.21)   small/medium cities
  T3  200 000 ≤ Pop < 1 000 000 (12.21–13.82) large cities
  T4  Pop ≥ 1 000 000       (ln P ≥ 13.82)  metropolitan cities

These thresholds are FIXED inputs from upstream — this module does NOT
re-derive or re-validate them.  The tiers are used only for
visualisation and stratified description; all formal inference uses
continuous log(Pop) via OLS (Script 08).

A supplementary two-breakpoint segmented regression on the Part 3
city-level H17 and r17 indicators is run here for descriptive context —
to show that these morphological indicators also exhibit non-linear
changes near the inherited tier boundaries.  The breakpoint estimates
(H17 ≈ 50k/1m, r17 ≈ 20k/2m) do not replace the Part 3 thresholds;
they only confirm that internal morphological gradients are also
non-linear in the same region of the population spectrum.

Outputs
-------
  outputs/tables/city_tiers.parquet      — one row per city, with tier + audit cols
  outputs/tables/tier_audit.csv          — N_cities / N_cells / median_pop / N_countries
                                           / median_R98_km per tier
  outputs/tables/tier_segmented_reg.csv  — breakpoint estimates (H17, r17)
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
PART3_DIR   = ROOT.parent.parent / "03_urban_hierarchy" / "outputs" / "tables"
PART4_DIR   = ROOT.parent / "core" / "outputs" / "tables"
OUT_TABLES  = ROOT / "outputs" / "tables"

CELL_PQ     = PART4_DIR / "cell_radial_position.parquet"
CITY_PQ     = PART3_DIR / "city_metrics.parquet"
R98_PQ      = PART4_DIR / "city_R98.parquet"
PARAMS_JSON = PART4_DIR / "allometric_params.json"

# ── tier definition ────────────────────────────────────────────────────────
TIER_BREAKS = [0, 20_000, 200_000, 1_000_000, np.inf]
TIER_LABELS = ["T1", "T2", "T3", "T4"]
TIER_NAMES  = {
    "T1": "Tier 1 (<20k)",
    "T2": "Tier 2 (20k–200k)",
    "T3": "Tier 3 (200k–1m)",
    "T4": "Tier 4 (≥1m)",
}
MIN_CELLS   = 50   # minimum cells for a city to enter analysis

# ── helpers ────────────────────────────────────────────────────────────────

def _assign_tier(pop_series: pd.Series) -> pd.Series:
    return pd.cut(pop_series, bins=TIER_BREAKS, labels=TIER_LABELS, right=False)


def _segmented_two_break(x: np.ndarray, y: np.ndarray,
                          candidates: np.ndarray) -> dict:
    """
    Grid-search two-breakpoint piecewise linear fit via OLS (3-segment).
    Returns breakpoint pair and R² for best fit.
    """
    from itertools import combinations
    best_r2  = -np.inf
    best_bp  = (np.nan, np.nan)
    n        = len(x)
    X1       = np.column_stack([np.ones(n), x])

    for bp1, bp2 in combinations(candidates, 2):
        if bp1 >= bp2:
            continue
        seg = np.zeros(n, dtype=int)
        seg[x >= bp1] = 1
        seg[x >= bp2] = 2
        Xd = np.column_stack([X1,
                               np.where(seg >= 1, x - bp1, 0),
                               np.where(seg == 2, x - bp2, 0)])
        try:
            coef, res, _, _ = np.linalg.lstsq(Xd, y, rcond=None)
            ss_res = float(np.sum((y - Xd @ coef) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            if r2 > best_r2:
                best_r2 = r2
                best_bp = (bp1, bp2)
        except np.linalg.LinAlgError:
            continue
    return {"bp1": best_bp[0], "bp2": best_bp[1], "r2": best_r2}


# ── main ───────────────────────────────────────────────────────────────────

def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    # ── load allometric params ───────────────────────────────────────────
    log.info("Loading allometric_params.json ...")
    with open(PARAMS_JSON) as f:
        params = json.load(f)
    alpha = float(params["alpha_R98"])
    log.info(f"  alpha_R98 = {alpha:.4f}")

    # ── load city-level data (Part 3) ────────────────────────────────────
    log.info("Loading city_metrics.parquet ...")
    city = pd.read_parquet(CITY_PQ,
                           columns=["city", "country", "log_pop",
                                    "n_cells", "community",
                                    "H17_star", "r17"])
    city = city.rename(columns={"H17_star": "H17_city", "r17": "r17_city"})
    city["pop"] = np.exp(city["log_pop"])
    log.info(f"  {len(city)} cities in Part 3.")

    # ── QC: minimum cell count ───────────────────────────────────────────
    city_qc = city[city["n_cells"] >= MIN_CELLS].copy()
    log.info(f"  After n_cells >= {MIN_CELLS}: {len(city_qc)} cities.")

    # ── assign tiers ─────────────────────────────────────────────────────
    city_qc["tier"] = _assign_tier(city_qc["pop"])
    log.info("Tier counts:")
    for t in TIER_LABELS:
        n = (city_qc["tier"] == t).sum()
        log.info(f"  {t} ({TIER_NAMES[t]}): {n} cities")

    # ── load R98 per city ────────────────────────────────────────────────
    log.info("Loading city_R98.parquet ...")
    r98 = pd.read_parquet(R98_PQ, columns=["city", "country", "R98_km"]) \
            .drop_duplicates(["city", "country"])
    city_qc = city_qc.merge(r98, on=["city", "country"], how="left")

    # ── load cell counts per city ────────────────────────────────────────
    log.info("Loading cell_radial_position.parquet (columns: city, country, Pop2022) ...")
    cells = pd.read_parquet(CELL_PQ, columns=["city", "country", "Pop2022"])
    cell_counts = cells.groupby(["city", "country"]).size().reset_index(name="n_cells_check")
    city_qc = city_qc.merge(cell_counts, on=["city", "country"], how="left")

    # ── save city_tiers ──────────────────────────────────────────────────
    out_cols = ["city", "country", "pop", "log_pop", "tier",
                "n_cells", "community", "R98_km",
                "H17_city", "r17_city"]
    out_cols = [c for c in out_cols if c in city_qc.columns]
    _path = OUT_TABLES / "city_tiers.parquet"
    city_qc[out_cols].to_parquet(_path, index=False)
    _chk = pd.read_parquet(_path)
    assert len(_chk) == len(city_qc), \
        f"Parquet read-back mismatch: wrote {len(city_qc)}, read {len(_chk)}"
    log.info(f"Saved + verified: city_tiers.parquet  ({len(city_qc)} rows)")

    # ── tier audit ───────────────────────────────────────────────────────
    audit_rows = []
    for t in TIER_LABELS:
        sub = city_qc[city_qc["tier"] == t]
        audit_rows.append({
            "tier":           t,
            "tier_name":      TIER_NAMES[t],
            "n_cities":       len(sub),
            "n_countries":    sub["country"].nunique(),
            "median_pop":     sub["pop"].median(),
            "p5_pop":         sub["pop"].quantile(0.05),
            "p95_pop":        sub["pop"].quantile(0.95),
            "median_R98_km":  sub["R98_km"].median() if "R98_km" in sub else np.nan,
            "median_H17":     sub["H17_city"].median() if "H17_city" in sub else np.nan,
            "median_r17":     sub["r17_city"].median() if "r17_city" in sub else np.nan,
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT_TABLES / "tier_audit.csv", index=False, float_format="%.4f")
    log.info("Saved: tier_audit.csv")
    print("\n=== Tier audit ===")
    print(audit.to_string(index=False))

    # ── segmented regression (validate breakpoints) ──────────────────────
    log.info("Segmented regression on H17 and r17 vs log(Pop) ...")
    df_seg = city_qc.dropna(subset=["log_pop"])
    x = df_seg["log_pop"].values
    candidates = np.log(np.array([10_000, 20_000, 50_000, 100_000,
                                   200_000, 500_000, 1_000_000,
                                   2_000_000, 5_000_000]))

    seg_rows = []
    for outcome in ["H17_city", "r17_city"]:
        if outcome not in df_seg.columns:
            continue
        y = df_seg[outcome].dropna().values
        x_ = x[~np.isnan(df_seg[outcome].values)]
        res = _segmented_two_break(x_, y, candidates)
        seg_rows.append({
            "outcome": outcome,
            "bp1_lnP": res["bp1"],
            "bp1_pop": int(np.exp(res["bp1"])) if not np.isnan(res["bp1"]) else np.nan,
            "bp2_lnP": res["bp2"],
            "bp2_pop": int(np.exp(res["bp2"])) if not np.isnan(res["bp2"]) else np.nan,
            "r2": res["r2"],
        })
        log.info(f"  {outcome}: bp1≈{int(np.exp(res['bp1'])//1000)}k  "
                 f"bp2≈{int(np.exp(res['bp2'])//1000)}k  R²={res['r2']:.3f}")

    if seg_rows:
        pd.DataFrame(seg_rows).to_csv(
            OUT_TABLES / "tier_segmented_reg.csv", index=False, float_format="%.4f")
        log.info("Saved: tier_segmented_reg.csv")

    log.info("Done.")


if __name__ == "__main__":
    main()
