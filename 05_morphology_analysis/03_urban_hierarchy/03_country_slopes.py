"""
03_country_slopes.py
====================
Country-level OLS slopes for the four morphological group shares,
H* (diversity) and r (assortativity) vs log(population).

Outputs
-------
Four main figures (one per group share p1–p4):
    Each shows country-specific OLS lines (coloured by macro-region)
    overlaid on the continental GAM curve + CI from Panel A.

One summary figure:
    Slope distribution across countries for all six outcomes (p1–p4, H*, r).
    Each outcome = one column; countries = dots, coloured by macro-region.
    Annotates "N/47 countries with positive slope" for H* and r.

outputs/tables/country_slopes.csv
    OLS slope, intercept, R², n, 95% CI and p-value (analytic, Student-t)
    per country × outcome. The figures below remain purely descriptive
    (one line per country, no uncertainty shown); the CI/p-value columns
    let the table itself support country-level inference if needed,
    without over-claiming precision that individual countries' small n
    (as low as min_cities) doesn't support.

Usage
-----
    python 03_country_slopes.py [--min_cities N]

    --min_cities   Minimum number of cities to include a country (default: 10)
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

ROOT    = Path(__file__).resolve().parent
METRICS = ROOT / "outputs" / "tables" / "city_metrics.parquet"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_TABLES = ROOT / "outputs" / "tables"

# ---------------------------------------------------------------------------
# Macro-region assignment
# ---------------------------------------------------------------------------
MACRO_REGION = {
    # North Africa
    "Algeria": "North Africa", "Egypt": "North Africa",
    "Libya": "North Africa", "Morocco": "North Africa",
    "Tunisia": "North Africa", "Sudan": "North Africa",
    "Mauritania": "North Africa",
    # West Africa
    "Nigeria": "West Africa", "Ghana": "West Africa",
    "Côte dIvoire": "West Africa", "Senegal": "West Africa",
    "Burkina Faso": "West Africa", "Mali": "West Africa",
    "Niger": "West Africa", "Guinea": "West Africa",
    "Benin": "West Africa", "Togo": "West Africa",
    "Sierra Leone": "West Africa", "Liberia": "West Africa",
    "Gambia": "West Africa", "Guinea-Bissau": "West Africa",
    "Cape Verde": "West Africa", "Mauritius": "West Africa",
    # East Africa
    "Ethiopia": "East Africa", "Tanzania": "East Africa",
    "Uganda": "East Africa", "Kenya": "East Africa",
    "Rwanda": "East Africa", "Burundi": "East Africa",
    "South Sudan": "East Africa", "Somalia": "East Africa",
    "Eritrea": "East Africa", "Djibouti": "East Africa",
    "Madagascar": "East Africa",
    # Central Africa
    "Democratic Republic of the Congo": "Central Africa",
    "Republic of Congo": "Central Africa",
    "Cameroon": "Central Africa",
    "Central African Republic": "Central Africa",
    "Angola": "Central Africa",
    "Gabon": "Central Africa",
    "Equatorial Guinea": "Central Africa",
    "Sao Tome and Principe": "Central Africa",
    # Southern Africa
    "South Africa": "Southern Africa", "Mozambique": "Southern Africa",
    "Zimbabwe": "Southern Africa", "Zambia": "Southern Africa",
    "Malawi": "Southern Africa", "Botswana": "Southern Africa",
    "Namibia": "Southern Africa", "Lesotho": "Southern Africa",
    "Swaziland": "Southern Africa",
}

REGION_COLORS = {
    "North Africa":   "#E6AA3A",
    "West Africa":    "#E05C3A",
    "East Africa":    "#4CAF82",
    "Central Africa": "#7B67EE",
    "Southern Africa":"#4A90D9",
}
REGION_ORDER = ["North Africa", "West Africa", "East Africa",
                "Central Africa", "Southern Africa"]

GROUP_NAMES = {
    "p1": "G1 (CFL) — Cropland-Fringe Low-Rise",
    "p2": "G2 (HSF) — Heterogeneous Sparse Fabric",
    "p3": "G3 (LVC) — Large-Volume Compact Fabric",
    "p4": "G4 (DSB) — Dense Small-Building Fabric",
    "H_star": "Diversity H*",
    "r":      "Assortativity r",
}
GROUP_YLABELS = {
    "p1": "Group share",
    "p2": "Group share",
    "p3": "Group share",
    "p4": "Group share",
    "H_star": "Normalised Shannon H*",
    "r":      "Newman assortativity r",
}

GAM_N_SPLINES = 6
MM = 1 / 25.4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ols_slope(x: np.ndarray, y: np.ndarray):
    """
    Return (slope, intercept, r2, n, ci_lo, ci_hi, pval) from per-country
    OLS; nan if insufficient data.

    CI uses the analytic OLS standard error (Student-t, n-2 df) — plain,
    unclustered, which is the right choice HERE specifically because each
    call already IS one country's own within-country regression (there is
    no further grouping structure left to cluster over, unlike the
    continent-wide models in 06/07/08 which cluster by country because
    they pool many countries together).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum()
    if n < 5:
        return np.nan, np.nan, np.nan, int(n), np.nan, np.nan, np.nan
    res = stats.linregress(x[mask], y[mask])
    t_crit = stats.t.ppf(0.975, df=n - 2)
    ci_lo  = res.slope - t_crit * res.stderr
    ci_hi  = res.slope + t_crit * res.stderr
    return res.slope, res.intercept, res.rvalue ** 2, int(n), ci_lo, ci_hi, res.pvalue


def fit_continental_gam(df: pd.DataFrame, y_col: str, x_grid: np.ndarray):
    x = df["log_pop"].values
    y = df[y_col].values
    mask = np.isfinite(x) & np.isfinite(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gam = LinearGAM(s(0, n_splines=GAM_N_SPLINES)).gridsearch(
            x[mask, None], y[mask], progress=False
        )
    return gam.predict(x_grid[:, None])


def pop_grid(df: pd.DataFrame, n: int = 200) -> np.ndarray:
    lo = np.nanpercentile(df["log_pop"], 2)
    hi = np.nanpercentile(df["log_pop"], 98)
    return np.linspace(lo, hi, n)


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Figure 1–4: per-group country lines + continental GAM
# ---------------------------------------------------------------------------
def plot_group_country_lines(
    df: pd.DataFrame,
    slopes_df: pd.DataFrame,
    y_col: str,
    min_cities: int,
    out_prefix: str,
):
    fig, ax = plt.subplots(figsize=(130 * MM, 100 * MM))

    x_grid = pop_grid(df)
    y_hat  = fit_continental_gam(df, y_col, x_grid)

    # --- Country OLS lines (background) ---
    countries = slopes_df[
        (slopes_df["outcome"] == y_col) &
        (slopes_df["n"] >= min_cities) &
        slopes_df["slope"].notna()
    ].copy()

    x_lo = np.nanpercentile(df["log_pop"], 2)
    x_hi = np.nanpercentile(df["log_pop"], 98)

    for _, row in countries.iterrows():
        region = MACRO_REGION.get(row["country"], "West Africa")
        col    = REGION_COLORS.get(region, "#888888")
        alpha  = 0.55 if row["n"] >= 30 else 0.30
        lw     = 0.8  if row["n"] >= 30 else 0.5

        # Country x-range: clip to country's own pop range ± small margin
        sub = df[df["country"] == row["country"]]["log_pop"].dropna()
        cx_lo = max(sub.min() - 0.3, x_lo)
        cx_hi = min(sub.max() + 0.3, x_hi)
        cx    = np.array([cx_lo, cx_hi])
        cy    = row["slope"] * cx + row["intercept"]

        ax.plot(cx, cy, color=col, lw=lw, alpha=alpha, zorder=2)

    # --- Continental GAM (foreground) ---
    ax.plot(x_grid, y_hat, color="black", lw=2.0, zorder=5, label="Africa (GAM)")

    # --- Zero line for reference ---
    if y_col in ("r",):
        ax.axhline(0, color="gray", lw=0.6, ls=":", zorder=1)

    # --- Region legend ---
    handles = [
        mlines.Line2D([], [], color=REGION_COLORS[r], lw=1.5, label=r)
        for r in REGION_ORDER if r in REGION_COLORS
    ]
    handles.append(mlines.Line2D([], [], color="black", lw=2.0, label="Africa (GAM)"))
    ax.legend(handles=handles, fontsize=5.5, frameon=False,
              loc="best", ncol=1)

    # Slope direction summary in subtitle
    pos = (countries["slope"] > 0).sum()
    tot = len(countries)
    ax.set_title(
        f"{GROUP_NAMES[y_col]}\n"
        f"{pos}/{tot} countries: positive slope",
        fontsize=8, pad=5,
    )
    ax.set_xlabel("log(population)", fontsize=7)
    ax.set_ylabel(GROUP_YLABELS[y_col], fontsize=7)
    ax.tick_params(labelsize=6.5)
    _despine(ax)

    fig.tight_layout()
    safe_col = y_col.replace("*", "star").replace(" ", "_")
    fig.savefig(FIG_DIR / f"{out_prefix}_{safe_col}.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{out_prefix}_{safe_col}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Country-lines figure saved: %s", y_col)


# ---------------------------------------------------------------------------
# Summary: slope distribution across countries (dot plot, 6 columns)
# ---------------------------------------------------------------------------
def plot_slope_summary(slopes_df: pd.DataFrame, min_cities: int, out_prefix: str):
    outcomes  = ["p1", "p2", "p3", "p4", "H_star", "r"]
    n_out     = len(outcomes)

    fig, axes = plt.subplots(1, n_out, figsize=(180 * MM, 90 * MM),
                             sharey=False)
    fig.subplots_adjust(wspace=0.35)

    for ax, y_col in zip(axes, outcomes):
        sub = slopes_df[
            (slopes_df["outcome"] == y_col) &
            (slopes_df["n"] >= min_cities) &
            slopes_df["slope"].notna()
        ].copy()

        # Sort by macro-region then slope
        sub["region"] = sub["country"].map(
            lambda c: MACRO_REGION.get(c, "West Africa"))
        sub["region_order"] = sub["region"].map(
            {r: i for i, r in enumerate(REGION_ORDER)})
        sub = sub.sort_values(["region_order", "slope"]).reset_index(drop=True)

        y_pos  = np.arange(len(sub))
        colors = [REGION_COLORS.get(MACRO_REGION.get(c, ""), "#888888")
                  for c in sub["country"]]

        ax.scatter(sub["slope"], y_pos, c=colors, s=8, zorder=3,
                   edgecolors="none", alpha=0.85)
        ax.axvline(0, color="gray", lw=0.7, ls="--", zorder=1)

        pos = (sub["slope"] > 0).sum()
        ax.set_title(
            f"{GROUP_NAMES[y_col].split('—')[-1].strip()}\n"
            f"{pos}/{len(sub)} positive",
            fontsize=6, pad=3,
        )
        ax.set_xlabel("OLS slope β", fontsize=6)
        ax.tick_params(labelsize=5.5)
        ax.set_yticks([])
        _despine(ax)

    # Shared region legend
    handles = [
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor=REGION_COLORS[r],
                      markersize=4, label=r)
        for r in REGION_ORDER
    ]
    fig.legend(handles=handles, fontsize=5.5, frameon=False,
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        "Country-level OLS slopes: morphology ~ log(population)",
        fontsize=8, y=1.01,
    )
    fig.savefig(FIG_DIR / f"{out_prefix}_slope_summary.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{out_prefix}_slope_summary.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Slope summary figure saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(min_cities: int = 10):
    logging.info("Loading city metrics ...")
    df = pd.read_parquet(METRICS)
    df = df.dropna(subset=["log_pop", "community", "p1", "p2", "p3", "p4",
                           "H_star"]).copy()
    df["community"] = df["community"].astype(int)
    logging.info("  %d cities.", len(df))

    outcomes = ["p1", "p2", "p3", "p4", "H_star", "r"]

    # --- Compute per-country OLS slopes ---
    logging.info("Computing country OLS slopes ...")
    records = []
    for country, grp in df.groupby("country"):
        for y_col in outcomes:
            sl, ic, r2, n, ci_lo, ci_hi, pval = ols_slope(
                grp["log_pop"].values, grp[y_col].values
            )
            records.append({
                "country":   country,
                "outcome":   y_col,
                "slope":     sl,
                "intercept": ic,
                "r2":        r2,
                "n":         n,
                "ci_lo":     ci_lo,
                "ci_hi":     ci_hi,
                "pval":      pval,
                "region":    MACRO_REGION.get(country, "Unknown"),
            })

    slopes_df = pd.DataFrame(records)
    slopes_df.to_csv(OUT_TABLES / "country_slopes.csv", index=False)
    logging.info("country_slopes.csv saved (%d rows).", len(slopes_df))

    # Quick summary
    for y_col in outcomes:
        sub = slopes_df[(slopes_df["outcome"] == y_col) &
                        (slopes_df["n"] >= min_cities) &
                        slopes_df["slope"].notna()]
        pos = (sub["slope"] > 0).sum()
        logging.info("  %s: %d/%d countries positive slope  "
                     "(median β=%.4f, range [%.4f, %.4f])",
                     y_col, pos, len(sub),
                     sub["slope"].median(),
                     sub["slope"].min(), sub["slope"].max())

    prefix = "country_hierarchy"

    # --- Per-group country-lines figures ---
    for y_col in outcomes:
        logging.info("Plotting country lines: %s ...", y_col)
        plot_group_country_lines(df, slopes_df, y_col, min_cities, prefix)

    # --- Slope summary ---
    logging.info("Plotting slope summary ...")
    plot_slope_summary(slopes_df, min_cities, prefix)

    logging.info("Done. All figures in %s", FIG_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Country-level hierarchy slopes for morphological groups."
    )
    parser.add_argument(
        "--min_cities", type=int, default=10,
        help="Minimum city count to include a country (default: 10).",
    )
    args = parser.parse_args()
    main(min_cities=args.min_cities)
