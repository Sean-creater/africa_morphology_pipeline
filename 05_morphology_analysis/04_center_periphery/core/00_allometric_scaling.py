"""
00_allometric_scaling.py
========================
Estimate allometric scaling of African cities: spatial extent vs population.

Computes R98 (98th-percentile radial distance from primary population centre)
for every city, then fits:

    log(R98) ~ log(Pop)   → scaling exponent α̂  (spatial radius)
    log(Area) ~ log(Pop)  → scaling exponent β̂  (urban area ~ n_cells)

These exponents parameterise the population-scaled distance used in 03–06:
    d_scaled = dist_m / Pop^α̂

Also determines appropriate absolute-km ring bands by inspecting the R98
distribution, and computes the fixed scaled-distance quintile bins over all
cells.

Outputs
-------
    outputs/tables/city_R98.parquet
        city, country, log_pop, n_cells, R98_m, log_R98

    outputs/tables/allometric_params.json
        {
          "alpha_R98":      α̂,
          "alpha_R98_ci":   [lo, hi],
          "beta_area":      β̂,
          "beta_area_ci":   [lo, hi],
          "R98_percentiles_km": {p5, p25, p50, p75, p95},
          "km_bins":        [0, 1, 3, 5, 10, inf],
          "scl_bins":       [q0, q20, q40, q60, q80, inf],
          "scl_median":     median_d_scaled
        }

    outputs/figures/allometric_scaling.png
        Two panels: R98 ~ Pop and Area ~ Pop log-log scatter + OLS fit.

Usage
-----
    python 00_allometric_scaling.py
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

ROOT         = Path(__file__).resolve().parent
OUT_DIR      = ROOT / "outputs" / "tables"
FIG_DIR      = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_POS_PATH  = OUT_DIR / "cell_radial_position.parquet"
CITY_METRICS   = (ROOT.parent.parent / "03_urban_hierarchy" /
                  "outputs" / "tables" / "city_metrics.parquet")

MM = 1 / 25.4

# Initial km band proposal (will be reviewed against data)
KM_BINS_PROPOSAL = [0.0, 1.0, 3.0, 5.0, 10.0, np.inf]


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#AAAAAA")
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(colors="#555555", labelsize=7)


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logging.info("Loading cell_radial_position.parquet ...")
    cell_pos = pd.read_parquet(CELL_POS_PATH,
                               columns=["city", "country", "dist_m_primary"])
    logging.info("  %d cells.", len(cell_pos))

    logging.info("Loading city_metrics.parquet ...")
    city_metrics = pd.read_parquet(CITY_METRICS,
                                   columns=["city", "country", "log_pop", "n_cells"])
    logging.info("  %d cities.", len(city_metrics))

    # ------------------------------------------------------------------
    # 2. Compute R98 per city
    # ------------------------------------------------------------------
    logging.info("Computing R98 (98th percentile dist_m_primary) per city ...")
    r98_df = (cell_pos.groupby(["city", "country"])["dist_m_primary"]
              .quantile(0.98).rename("R98_m").reset_index())
    logging.info("  %d cities with R98.", len(r98_df))

    # Merge with city metrics
    r98_df = r98_df.merge(city_metrics, on=["city", "country"], how="inner")
    logging.info("  %d cities after merge with city_metrics.", len(r98_df))

    # Drop degenerate cases
    r98_df = r98_df[r98_df["R98_m"] > 10].copy()   # at least 10 m radius
    r98_df = r98_df[r98_df["log_pop"].notna() & r98_df["n_cells"].notna()].copy()
    logging.info("  %d cities after QC (R98 > 10m).", len(r98_df))

    r98_df["log_R98"]    = np.log(r98_df["R98_m"])
    r98_df["R98_km"]     = r98_df["R98_m"] / 1000.0
    r98_df["log_n_cells"] = np.log(r98_df["n_cells"])

    # ------------------------------------------------------------------
    # 3. R98 distribution
    # ------------------------------------------------------------------
    pcts = {
        "p5":  float(np.percentile(r98_df["R98_km"], 5)),
        "p25": float(np.percentile(r98_df["R98_km"], 25)),
        "p50": float(np.percentile(r98_df["R98_km"], 50)),
        "p75": float(np.percentile(r98_df["R98_km"], 75)),
        "p95": float(np.percentile(r98_df["R98_km"], 95)),
    }
    logging.info("R98 distribution (km): p5=%.2f  p25=%.2f  p50=%.2f  p75=%.2f  p95=%.2f",
                 pcts["p5"], pcts["p25"], pcts["p50"], pcts["p75"], pcts["p95"])

    # How many cities reach each km threshold?
    for km in [1, 3, 5, 10]:
        n = (r98_df["R98_km"] >= km).sum()
        logging.info("  Cities with R98 >= %d km: %d (%.1f%%)",
                     km, n, 100 * n / len(r98_df))

    # Cities covered in each abs band
    km_bins = KM_BINS_PROPOSAL
    for b0, b1 in zip(km_bins[:-1], km_bins[1:]):
        b1_str = f"{b1:.0f}" if np.isfinite(b1) else "∞"
        n = ((r98_df["R98_km"] > b0)).sum()  # cities that reach at least b0
        logging.info("  Cities with R98 > %.0f km (cover band %s-%s): %d",
                     b0, b0, b1_str, n)

    # ------------------------------------------------------------------
    # 4. OLS: log(R98) ~ log(Pop)   [country-clustered SE]
    # ------------------------------------------------------------------
    logging.info("OLS: log(R98) ~ log(Pop) (country-clustered SE) ...")
    res_r = smf.ols("log_R98 ~ log_pop", data=r98_df).fit(
        cov_type="cluster", cov_kwds={"groups": r98_df["country"]})
    alpha_hat = float(res_r.params["log_pop"])
    ci_r      = res_r.conf_int().loc["log_pop"]
    alpha_ci  = [float(ci_r[0]), float(ci_r[1])]
    logging.info("  α̂ = %.4f  [%.4f, %.4f]  R²=%.3f  n=%d",
                 alpha_hat, alpha_ci[0], alpha_ci[1],
                 res_r.rsquared, int(res_r.nobs))

    # ------------------------------------------------------------------
    # 5. OLS: log(Area=n_cells) ~ log(Pop)
    # ------------------------------------------------------------------
    logging.info("OLS: log(n_cells) ~ log(Pop) (country-clustered SE) ...")
    res_a = smf.ols("log_n_cells ~ log_pop", data=r98_df).fit(
        cov_type="cluster", cov_kwds={"groups": r98_df["country"]})
    beta_hat = float(res_a.params["log_pop"])
    ci_a     = res_a.conf_int().loc["log_pop"]
    beta_ci  = [float(ci_a[0]), float(ci_a[1])]
    logging.info("  β̂ = %.4f  [%.4f, %.4f]  R²=%.3f  n=%d",
                 beta_hat, beta_ci[0], beta_ci[1],
                 res_a.rsquared, int(res_a.nobs))

    # ------------------------------------------------------------------
    # 6. Population-scaled distance: compute bins from full cell data
    # ------------------------------------------------------------------
    logging.info("Computing population-scaled distance bins ...")
    # Merge log_pop back to cell_pos for scaled-distance computation
    cell_pos2 = cell_pos.merge(
        city_metrics[["city", "country", "log_pop"]], on=["city", "country"], how="inner")
    cell_pos2 = cell_pos2[cell_pos2["log_pop"].notna()].copy()
    cell_pos2["pop"] = np.exp(cell_pos2["log_pop"])
    # d_scaled = dist_m / pop^alpha  (units: metres / person^alpha)
    cell_pos2["d_scaled"] = (cell_pos2["dist_m_primary"].astype("float64") /
                              np.power(cell_pos2["pop"], alpha_hat))
    cell_pos2 = cell_pos2[cell_pos2["d_scaled"].notna() &
                           np.isfinite(cell_pos2["d_scaled"]) &
                           (cell_pos2["d_scaled"] > 0)].copy()
    logging.info("  %d cells with valid d_scaled.", len(cell_pos2))

    # 5-quantile bins (quintiles) fixed over all cells
    # Use 0, 20th, 40th, 60th, 80th percentile as interior edges; last bin is open
    qs = np.percentile(cell_pos2["d_scaled"], [20, 40, 60, 80])
    scl_bins = [0.0] + [float(q) for q in qs] + [np.inf]

    # Verify: all cells get a ring 1..5
    # Use interior edges (exclude 0 and inf) for digitize
    interior_edges = scl_bins[1:-1]   # [q20, q40, q60, q80]
    scl_ring_test = np.digitize(cell_pos2["d_scaled"].values, interior_edges) + 1
    scl_ring_test = np.clip(scl_ring_test, 1, 5)
    logging.info("  Scaled ring distribution (test): %s",
                 dict(zip(*np.unique(scl_ring_test, return_counts=True))))

    scl_median = float(np.median(cell_pos2["d_scaled"]))
    logging.info("  Median d_scaled = %.6g  (scl_bins: %s)",
                 scl_median, [f"{b:.4g}" for b in scl_bins])

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    r98_df[["city","country","log_pop","n_cells","R98_m","log_R98","R98_km"]].to_parquet(
        OUT_DIR / "city_R98.parquet", index=False)
    logging.info("Saved: city_R98.parquet")

    params = {
        "alpha_R98":             alpha_hat,
        "alpha_R98_ci":          alpha_ci,
        "beta_area":             beta_hat,
        "beta_area_ci":          beta_ci,
        "R98_percentiles_km":    pcts,
        "km_bins":               [float(b) for b in km_bins],
        "scl_bins":              [float(b) for b in scl_bins],
        "scl_median":            scl_median,
    }
    with open(OUT_DIR / "allometric_params.json", "w") as f:
        json.dump(params, f, indent=2)
    logging.info("Saved: allometric_params.json")

    # ------------------------------------------------------------------
    # 8. Figure: two-panel allometric log-log scatter
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(160 * MM, 75 * MM))
    fig.patch.set_facecolor("white")

    for ax, (res, y_col, y_lab, exponent, exp_label) in zip(
        axes,
        [
            (res_r, "log_R98",     "log R₉₈ (m)",     alpha_hat, "α̂"),
            (res_a, "log_n_cells", "log n_cells",      beta_hat,  "β̂"),
        ],
    ):
        x = r98_df["log_pop"].values
        y = r98_df[y_col].values
        ax.scatter(x, y, s=3, alpha=0.25, color="#AAAAAA", rasterized=True)

        # OLS fit line
        x_fit = np.linspace(x.min(), x.max(), 200)
        ic    = float(res.params["Intercept"])
        sl    = exponent
        ax.plot(x_fit, ic + sl * x_fit, color="#2166AC", lw=1.6)

        ax.set_xlabel("log(Population)", fontsize=8)
        ax.set_ylabel(y_lab, fontsize=8)
        ci = res.conf_int().loc["log_pop"]
        ax.set_title(
            f"{exp_label} = {exponent:.3f} [{float(ci[0]):.3f}, {float(ci[1]):.3f}]"
            f"   R² = {res.rsquared:.3f}",
            fontsize=7.5, pad=4
        )
        ax.tick_params(labelsize=6.5)
        _despine(ax)

    plt.tight_layout(pad=0.8)
    fig.savefig(FIG_DIR / "allometric_scaling.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Saved: allometric_scaling.png")

    # ------------------------------------------------------------------
    # 9. Summary printout for expert review
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("ALLOMETRIC SCALING RESULTS")
    print("="*60)
    print(f"  Cities analysed:  {int(res_r.nobs)}")
    print(f"  α̂ (R98 ~ Pop):   {alpha_hat:.4f}  95% CI [{alpha_ci[0]:.4f}, {alpha_ci[1]:.4f}]"
          f"   R²={res_r.rsquared:.3f}")
    print(f"  β̂ (Area ~ Pop):  {beta_hat:.4f}  95% CI [{beta_ci[0]:.4f}, {beta_ci[1]:.4f}]"
          f"   R²={res_a.rsquared:.3f}")
    print(f"\n  R98 distribution (km):")
    for k, v in pcts.items():
        print(f"    {k}: {v:.2f} km")
    print(f"\n  km_bins (absolute distance rings): {km_bins}")
    print(f"  scl_bins (pop-scaled distance):    {[f'{b:.4g}' for b in scl_bins]}")
    print(f"  scl_median (d_scaled):             {scl_median:.4g}")
    print("="*60)

    logging.info("Done.")


if __name__ == "__main__":
    main()
