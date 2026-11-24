"""
07_size_gradient.py
===================
Two publication figures (H17_star and r17) showing the continent-wide size
gradient: grey scatter + OLS regression line (dashed) + LOESS smooth (solid),
each with 95% CI bands. Standardised β and R² annotated on the figure.

These are the main-text figures and are kept to just the headline
relationship — no robustness inset. The full robustness picture for both
outcomes (raw / community FE / country FE / size-controlled / rarefied /
permutation-null z for r17) lives in the supplementary figures produced
by 06_scaling_regression.py (scaling_coef.svg) and 08_robustness.py
(robustness_coef.svg), which already include every comparison needed —
notably r17_z, the methodologically correct test of whether r17's
population relationship is a graph-size artefact (it isn't: a naive
+log(n_edges) covariate control flips the sign, but that control itself
over-corrects — n_edges mixes cell count, graph topology AND
population-driven spatial extent, so it's a conditional association, not
confound removal — and the permutation-null z-score, which holds each
city's own graph structure and type-count distribution fixed, confirms
the original positive relationship, even more strongly).

Output
------
    outputs/figures/size_gradient_H17_star.pdf/png
    outputs/figures/size_gradient_r17.pdf/png

Usage
-----
    python 07_size_gradient.py [--boots N] [--workers N]
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


matplotlib.rcParams.update({
    "font.family":     "Arial",
    "font.size":       8,
    "svg.fonttype": "none",
})

ROOT      = Path(__file__).resolve().parent
METRICS   = ROOT / "outputs" / "tables" / "city_metrics.parquet"
FIG_DIR   = ROOT / "outputs" / "figures"
CACHE_DIR = ROOT / "outputs" / "cache"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4

LOESS_FRAC     = 0.35
N_BOOTS        = 500
N_BOOT_WORKERS = 14

OUTCOME_CFG = {
    "H17_star": dict(
        title      = "Morphological diversity H* (k = 17)",
        y_label    = "Normalised Shannon diversity H*",
        y_lim      = (-0.02, 1.02),
        color      = "#2166AC",
        y_major    = 0.2,
        y_minor    = 0.1,
        legend_loc = "upper left",
    ),
    "r17": dict(
        title      = "Spatial assortativity r (k = 17)",
        y_label    = "Newman assortativity r",
        y_lim      = (-0.15, 0.55),
        color      = "#D6604D",
        y_major    = 0.1,
        y_minor    = 0.05,
        legend_loc = "upper left",
    ),
}


# ---------------------------------------------------------------------------
# LOESS + country-cluster bootstrap CI (cached)
# ---------------------------------------------------------------------------
def fit_loess(x, y, x_grid, frac=LOESS_FRAC):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.full(len(x_grid), np.nan)
    sm = lowess(y[mask], x[mask], frac=frac, return_sorted=True)
    return np.interp(x_grid, sm[:, 0], sm[:, 1])


def _boot_worker(args):
    seed, x_vals, y_vals, countries, x_grid, frac = args
    rng      = np.random.default_rng(seed)
    unique_c = np.unique(countries)
    chosen   = rng.choice(unique_c, size=len(unique_c), replace=True)
    idx      = np.concatenate([np.where(countries == c)[0] for c in chosen])
    xb, yb   = x_vals[idx], y_vals[idx]
    mask     = np.isfinite(xb) & np.isfinite(yb)
    if mask.sum() < 20:
        return None
    try:
        sm = lowess(yb[mask], xb[mask], frac=frac, return_sorted=True)
        return np.interp(x_grid, sm[:, 0], sm[:, 1])
    except Exception:
        return None


def loess_ci(df_sub, y_col, x_grid, n_boots, n_workers, frac, cache_key=None):
    """LOESS + country-cluster bootstrap CI. Results cached to disk.

    The cache is invalidated whenever METRICS' mtime changes, so a
    regenerated city_metrics.parquet always triggers a fresh bootstrap
    rather than silently reusing a CI computed from stale input data.
    """
    x   = df_sub["log_pop"].values
    y   = df_sub[y_col].values
    cty = df_sub["country"].values
    y_hat = fit_loess(x, y, x_grid, frac=frac)
    input_mtime = METRICS.stat().st_mtime

    if cache_key:
        cache_file = CACHE_DIR / f"{cache_key}.npz"
        if cache_file.exists():
            cached = np.load(cache_file)
            if float(cached.get("input_mtime", -1)) == input_mtime:
                logging.info("  Loaded cached CI: %s", cache_key)
                return y_hat, cached["ci_lo"], cached["ci_hi"]
            logging.info("  Cache stale for %s (input changed), recomputing ...", cache_key)

    seeds = [int(sq.generate_state(1)[0])
             for sq in np.random.SeedSequence(42).spawn(n_boots)]
    args  = [(sd, x, y, cty, x_grid, frac) for sd in seeds]
    preds = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(_boot_worker, args):
            if res is not None:
                preds.append(res)
    preds = np.array(preds)
    ci_lo = np.percentile(preds, 2.5,  axis=0)
    ci_hi = np.percentile(preds, 97.5, axis=0)

    if cache_key:
        np.savez(cache_file, ci_lo=ci_lo, ci_hi=ci_hi, input_mtime=input_mtime)
        logging.info("  Cached: %s", cache_key)

    return y_hat, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# OLS regression line + 95% confidence band
# ---------------------------------------------------------------------------
def ols_line(df_sub, y_col, x_grid):
    """Return (y_hat, ci_lo, ci_hi) from OLS on the x_grid, country-clustered SE."""
    df_fit = df_sub[["log_pop", y_col, "country", "community"]].dropna().copy()
    # M1 raw
    m1 = smf.ols(f"{y_col} ~ log_pop", data=df_fit).fit(
        cov_type="cluster", cov_kwds={"groups": df_fit["country"]}
    )
    alpha, beta = m1.params["Intercept"], m1.params["log_pop"]
    y_line = alpha + beta * x_grid

    # 95% CI for the mean response, from the SAME country-clustered model
    # fit above (m1) — not a separately-fit plain-OLS model. statsmodels'
    # get_prediction() honours whatever cov_type the model was fit with,
    # so reusing m1 keeps the plotted band consistent with the clustered
    # beta/significance reported alongside it. A previous version fit a
    # second, unclustered OLS just for this band, which is near-certainly
    # too narrow here (47 country clusters vs ~6800+ cities — clustering
    # almost always widens SEs relative to treating every city as
    # independent).
    # m1 was fit via the formula API (smf.ols), so get_prediction() needs
    # a DataFrame with a matching "log_pop" column, not a raw
    # constant-augmented array (that's the non-formula OLS API's format).
    X_pred_df = pd.DataFrame({"log_pop": x_grid})
    pred      = m1.get_prediction(X_pred_df)
    ci        = pred.conf_int(alpha=0.05)
    ci_lo, ci_hi = ci[:, 0], ci[:, 1]

    # Extract standardized β and R² from country-clustered model
    mu_x = df_fit["log_pop"].mean(); sd_x = df_fit["log_pop"].std()
    mu_y = df_fit[y_col].mean();     sd_y = df_fit[y_col].std()
    df_fit["X_z"] = (df_fit["log_pop"] - mu_x) / sd_x
    df_fit["Y_z"] = (df_fit[y_col]     - mu_y) / sd_y
    mz = smf.ols("Y_z ~ X_z", data=df_fit).fit(
        cov_type="cluster", cov_kwds={"groups": df_fit["country"]}
    )
    mz_fe = smf.ols("Y_z ~ X_z + C(community)", data=df_fit).fit(
        cov_type="cluster", cov_kwds={"groups": df_fit["country"]}
    )
    stats = dict(
        beta_raw   = mz.params["X_z"],
        ci_lo_raw  = mz.conf_int().loc["X_z", 0],
        ci_hi_raw  = mz.conf_int().loc["X_z", 1],
        r2_raw     = mz.rsquared,
        beta_fe    = mz_fe.params["X_z"],
        ci_lo_fe   = mz_fe.conf_int().loc["X_z", 0],
        ci_hi_fe   = mz_fe.conf_int().loc["X_z", 1],
        r2_fe      = mz_fe.rsquared,
    )
    return y_line, ci_lo, ci_hi, stats



# ---------------------------------------------------------------------------
# Axes style
# ---------------------------------------------------------------------------
def _style_ax(ax):
    ax.set_facecolor("white")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#AAAAAA")
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(colors="#555555", labelsize=7)
    ax.xaxis.label.set_color("#555555")
    ax.yaxis.label.set_color("#555555")


# ---------------------------------------------------------------------------
# One figure
# ---------------------------------------------------------------------------
def plot_gradient(df, y_col, cfg, n_boots, n_workers):
    df_sub = df.dropna(subset=["log_pop", y_col, "country", "community"]).copy()
    color  = cfg["color"]

    x_lo   = np.nanpercentile(df_sub["log_pop"], 1)
    x_hi   = np.nanpercentile(df_sub["log_pop"], 99.5)
    x_grid = np.linspace(x_lo, x_hi, 400)

    fig, ax = plt.subplots(figsize=(130 * MM, 100 * MM))
    fig.patch.set_facecolor("white")

    # --- Scatter (all grey, rasterized) ---
    ax.scatter(
        df_sub["log_pop"], df_sub[y_col],
        c="#CCCCCC", s=3, alpha=0.5,
        edgecolors="none", linewidths=0, rasterized=True, zorder=2,
    )

    # --- OLS line + CI band (behind LOESS) ---
    logging.info("  Running OLS for %s ...", y_col)
    y_ols, ols_lo, ols_hi, stats = ols_line(df_sub, y_col, x_grid)
    ax.fill_between(x_grid, ols_lo, ols_hi,
                    color=color, alpha=0.15, linewidth=0, zorder=3)
    ax.plot(x_grid, y_ols, color=color, lw=1.4,
            linestyle="--", zorder=4)

    # --- Continental LOESS + bootstrap CI (on top) ---
    logging.info("  Computing continental LOESS + CI for %s ...", y_col)
    cache_key = f"continent_{y_col}_b{n_boots}"
    y_hat, ci_lo, ci_hi = loess_ci(
        df_sub, y_col, x_grid, n_boots, n_workers,
        frac=LOESS_FRAC, cache_key=cache_key,
    )
    ax.fill_between(x_grid, ci_lo, ci_hi,
                    color=color, alpha=0.25, linewidth=0, zorder=5)
    ax.plot(x_grid, y_hat, color=color, lw=2.0, zorder=6)

    # --- Axes ---
    ax.set_xlim(x_lo - 0.1, x_hi + 0.8)  # extra right margin for β label
    ax.set_ylim(cfg["y_lim"])
    ax.set_title(cfg["title"], fontsize=9, pad=6, fontweight="bold")
    ax.set_xlabel("log(population)", fontsize=8)
    ax.set_ylabel(cfg["y_label"], fontsize=8)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(cfg["y_major"]))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(cfg["y_minor"]))
    ax.tick_params(which="major", labelsize=7, length=4)
    ax.tick_params(which="minor", length=2, labelbottom=False, labelleft=False)
    _style_ax(ax)

    # --- Simple legend: solid = LOESS, dashed = OLS ---
    h_loess = mlines.Line2D([], [], color=color, lw=2.0, label="LOESS")
    h_ols   = mlines.Line2D([], [], color=color, lw=1.4, ls="--", label="OLS")
    ax.legend(handles=[h_loess, h_ols], fontsize=7, frameon=False,
              loc=cfg["legend_loc"], handlelength=1.8)

    # --- β annotation: large text next to the OLS line, at right end ---
    # Place label at the right edge of the OLS line
    x_annot = x_grid[-1]
    y_annot = y_ols[-1]
    ax.text(x_annot + 0.05, y_annot,
            f"β = {stats['beta_raw']:.2f},  R² = {stats['r2_raw']:.2f}",
            fontsize=9, fontweight="bold", color=color,
            va="center", ha="left")

    fig.tight_layout(pad=1.2)
    safe = y_col.replace("*", "star")
    for ext in ("svg", "png"):
        out = FIG_DIR / f"size_gradient_{safe}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Saved: size_gradient_%s.pdf/png", safe)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_boots=N_BOOTS, n_workers=N_BOOT_WORKERS):
    logging.info("Loading city metrics ...")
    df = pd.read_parquet(METRICS)
    df = df.dropna(subset=["log_pop", "community"]).copy()
    df["community"] = df["community"].astype(int)
    logging.info("  %d cities loaded.", len(df))

    for y_col, cfg in OUTCOME_CFG.items():
        logging.info("=== %s ===", y_col)
        plot_gradient(df, y_col, cfg, n_boots=n_boots, n_workers=n_workers)

    logging.info("Done. Figures in %s", FIG_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boots",   type=int, default=N_BOOTS)
    parser.add_argument("--workers", type=int, default=N_BOOT_WORKERS)
    args = parser.parse_args()
    main(n_boots=args.boots, n_workers=args.workers)
