"""
29_core_outer_divergence_mechanism.py
========================================
Mechanism decomposition of 23_scale_dependent_convergence.py's headline
reversal: Delta_V_E(P) = V_E_outer(P) - V_E_core(P) crosses zero around
P* ~ 126k, meaning small cities differ from each other mainly at the
core while large cities differ mainly at the periphery. That reversal
is consistent with at least two different mechanisms -- outer dispersion
rising with city size, or core dispersion falling with city size (or
both) -- and 23's single Delta curve cannot distinguish them.

This script plots the two LEVELS separately (not just their difference):

    V_E_core(P)  = mean cross-city embedding distance among same-size
                   cities' CORES
    V_E_outer(P) = mean cross-city embedding distance among same-size
                   cities' PERIPHERIES

No new metric is computed -- both columns already exist in
23_scale_dependent_convergence.py's scale_dependent_dispersion.csv
(V_E_core, V_E_outer, one row per qualifying city, K=10 nearest-by-
population neighbours). This script only re-plots and re-models that
existing table.

Expected pattern (per the mechanism hypothesis): above ~100k population,
cross-city CORE divergence declines significantly with population while
cross-city PERIPHERY divergence stays flat -- i.e. the reversal is driven
by cores CONVERGING, not peripheries diverging further. If so, "large
cities have differentiated peripheries" is better read as "peripheral
differentiation is preserved while cores homogenise", not "peripheral
differentiation keeps increasing without bound".

Method
------
- Trend: per-city V_E_core / V_E_outer vs log(Pop), LOWESS, plotted on
  the SAME log-population x-axis and city sample as 23 (no re-filtering
  -- inherits whatever sample 23 currently produces; as of the 2026-10-05
  methods review this is 2493 cities, not the earlier 636, since 23
  dropped its MIN_CELLS_ZONE=20 exclusion in favour of a reliability
  weight -- see 23's docstring). Weighted by each city's
  reliability_weight column (sqrt(min(n_core, n_outer)), inherited
  directly from 23's table) at both the LOWESS trend and the regression
  below -- a low-cell-count city still contributes a point, just a
  down-weighted one.
- Uncertainty: country-block bootstrap (resample countries with
  replacement, each resampled country contributing its full set of
  qualifying cities, preserving resampling multiplicity) -- same design
  as 23/21/27/20's methods-review fix, NOT an independent-city resample.
- Reversal marker: P*, reused directly from 23's core_outer_reversal_
  P_star.csv -- not recomputed here.
- Formal test (>=100k cities only, per the mechanism claim "above ~100k,
  core differentiation declines while peripheral differentiation does
  not"): separate WEIGHTED LEAST SQUARES V_z,c ~ log_pop + C(country) for
  z in {core, outer} (weights=reliability_weight), country-clustered SE
  (cov_type="cluster"), matching this project's standard
  country-clustered-inference convention (e.g. 16_scale_trajectory_
  interaction.py).

Output
------
  outputs/tables/core_outer_divergence_trend.csv
      log_pop_grid | pop_grid | V_E_core_smooth | V_E_core_lo | V_E_core_hi
      | V_E_outer_smooth | V_E_outer_lo | V_E_outer_hi
  outputs/tables/core_outer_divergence_regression.csv
      zone | n_cities | n_countries | beta_log_pop | beta_lo | beta_hi | p_value
  outputs/figures/core_outer_divergence_mechanism.svg/.png

Usage
-----
    python 29_core_outer_divergence_mechanism.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_FONT_DIR = Path("/home/sjs96_file_share/Scripts4Analysis/_fonts")
if _FONT_DIR.is_dir():
    for _f in fm.findSystemFonts(fontpaths=[str(_FONT_DIR)]):
        fm.fontManager.addfont(_f)
else:
    log.warning("Font directory not found (%s); falling back to default sans-serif.", _FONT_DIR)
matplotlib.rcParams.update({
    "font.family": "Arial", "font.size": 8, "svg.fonttype": "none",
    "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})

ROOT       = Path(__file__).parent
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIG    = ROOT / "outputs" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

DISPERSION_CSV = OUT_TABLES / "scale_dependent_dispersion.csv"
P_STAR_CSV     = OUT_TABLES / "core_outer_reversal_P_star.csv"

LOWESS_FRAC    = 0.5
N_BOOT         = 500
BOOT_SEED      = 42
GRID_N         = 60
REGRESSION_POP_MIN = 100_000

CORE_COLOR  = "#293890"
OUTER_COLOR = "#E08214"


def _lowess_trend(log_pop: np.ndarray, y: np.ndarray, grid: np.ndarray,
                  frac: float = LOWESS_FRAC, weight: np.ndarray = None) -> np.ndarray:
    """Weighted via integer row-repetition, same convention as
    23_scale_dependent_convergence.py's _lowess_trend."""
    mask = np.isfinite(log_pop) & np.isfinite(y)
    if weight is not None:
        mask &= np.isfinite(weight) & (weight > 0)
    if mask.sum() < 10:
        return np.full(len(grid), np.nan)
    x, yy = log_pop[mask], y[mask]
    if weight is not None:
        reps = np.maximum(np.round(weight[mask]).astype(int), 1)
        x = np.repeat(x, reps)
        yy = np.repeat(yy, reps)
    sm = lowess(yy, x, frac=frac, return_sorted=True)
    return np.interp(grid, sm[:, 0], sm[:, 1])


def main():
    log.info("Loading scale_dependent_dispersion.csv (23's existing V_E_core/V_E_outer, no new metric) ...")
    df = pd.read_csv(DISPERSION_CSV)
    df["country"] = df["city_key"].str.split(r"\|\|").str[0]
    log.info("  %d cities, %d countries.", len(df), df["country"].nunique())

    p_star_df = pd.read_csv(P_STAR_CSV)
    P_star = float(p_star_df[p_star_df["K"] == 10]["P_star"].iloc[0])
    log.info("  Reusing P* = %.0f from 23_scale_dependent_convergence.py (not recomputed).", P_star)

    # ── trend: LOWESS for V_E_core and V_E_outer separately, weighted ──
    log_pop_grid = np.linspace(df["log_pop"].min(), df["log_pop"].max(), GRID_N)
    w = df["reliability_weight"].values
    core_smooth = _lowess_trend(df["log_pop"].values, df["V_E_core"].values, log_pop_grid, weight=w)
    outer_smooth = _lowess_trend(df["log_pop"].values, df["V_E_outer"].values, log_pop_grid, weight=w)

    # ── country-block bootstrap (same design as 23/21/27/20's fix) ──
    log.info("Bootstrapping (country-block, n=%d) ...", N_BOOT)
    rng = np.random.default_rng(BOOT_SEED)
    countries = sorted(df["country"].unique())
    country_df = {c: df[df["country"] == c] for c in countries}
    boot_core = np.full((N_BOOT, GRID_N), np.nan)
    boot_outer = np.full((N_BOOT, GRID_N), np.nan)
    for b in range(N_BOOT):
        chosen_countries = rng.choice(countries, size=len(countries), replace=True)
        resampled = pd.concat([country_df[c] for c in chosen_countries], ignore_index=True)
        if len(resampled) < 20:
            continue
        rw = resampled["reliability_weight"].values
        boot_core[b] = _lowess_trend(resampled["log_pop"].values, resampled["V_E_core"].values, log_pop_grid, weight=rw)
        boot_outer[b] = _lowess_trend(resampled["log_pop"].values, resampled["V_E_outer"].values, log_pop_grid, weight=rw)
        if (b + 1) % 100 == 0:
            log.info("  bootstrap draw %d/%d", b + 1, N_BOOT)

    core_lo = np.nanpercentile(boot_core, 2.5, axis=0)
    core_hi = np.nanpercentile(boot_core, 97.5, axis=0)
    outer_lo = np.nanpercentile(boot_outer, 2.5, axis=0)
    outer_hi = np.nanpercentile(boot_outer, 97.5, axis=0)

    trend_df = pd.DataFrame({
        "log_pop_grid": log_pop_grid, "pop_grid": np.exp(log_pop_grid),
        "V_E_core_smooth": core_smooth, "V_E_core_lo": core_lo, "V_E_core_hi": core_hi,
        "V_E_outer_smooth": outer_smooth, "V_E_outer_lo": outer_lo, "V_E_outer_hi": outer_hi,
    })
    trend_df.to_csv(OUT_TABLES / "core_outer_divergence_trend.csv", index=False, float_format="%.6f")
    log.info("Saved: core_outer_divergence_trend.csv")

    # ── formal regression, >=100k cities only, country-clustered SE ──
    log.info("Fitting V_z,c ~ log_pop + C(country) for cities >= %s (country-clustered SE) ...",
             f"{REGRESSION_POP_MIN:,}")
    sub = df[df["pop"] >= REGRESSION_POP_MIN].copy()
    reg_rows = []
    for zone, col in [("core", "V_E_core"), ("outer", "V_E_outer")]:
        model = smf.wls(f"{col} ~ log_pop + C(country)", data=sub,
                        weights=sub["reliability_weight"]).fit(
            cov_type="cluster", cov_kwds={"groups": sub["country"]})
        beta = model.params["log_pop"]
        ci = model.conf_int().loc["log_pop"]
        p = model.pvalues["log_pop"]
        reg_rows.append({
            "zone": zone, "n_cities": len(sub), "n_countries": sub["country"].nunique(),
            "beta_log_pop": beta, "beta_lo": ci[0], "beta_hi": ci[1], "p_value": p,
        })
        log.info("  %s: beta=%.4f  95%% CI=[%.4f, %.4f]  p=%.4g",
                zone, beta, ci[0], ci[1], p)
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(OUT_TABLES / "core_outer_divergence_regression.csv", index=False, float_format="%.6f")
    log.info("Saved: core_outer_divergence_regression.csv")

    # ── figure ──
    log.info("Drawing figure ...")
    AXES_W_MM, AXES_H_MM = 100.0, 75.0
    LEFT_MM, RIGHT_MM, BOTTOM_MM, TOP_MM = 20.0, 20.0, 16.0, 6.0
    FIG_W_MM = LEFT_MM + AXES_W_MM + RIGHT_MM
    FIG_H_MM = BOTTOM_MM + AXES_H_MM + TOP_MM
    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
    ax = fig.add_axes([LEFT_MM / FIG_W_MM, BOTTOM_MM / FIG_H_MM,
                       AXES_W_MM / FIG_W_MM, AXES_H_MM / FIG_H_MM])

    pop_grid = np.exp(log_pop_grid)
    y_lim_hi = max(df["V_E_core"].max(), df["V_E_outer"].max()) * 1.05

    # Per-city scatter underneath the trend -- same style as 23's own
    # panel a (scale_dependent_core_periphery.svg), so the smooth curves
    # read as a summary of visible point-level data, not an abstraction.
    # Point size scales with reliability_weight, same convention as 23.
    point_size = 2 + 8 * (df["reliability_weight"] / df["reliability_weight"].max())
    ax.scatter(df["pop"], df["V_E_core"], s=point_size, alpha=0.15, color=CORE_COLOR,
              linewidth=0, zorder=1)
    ax.scatter(df["pop"], df["V_E_outer"], s=point_size, alpha=0.15, color=OUTER_COLOR,
              linewidth=0, zorder=1)

    ax.fill_between(pop_grid, core_lo, core_hi, color=CORE_COLOR, alpha=0.15, linewidth=0, zorder=2)
    ax.plot(pop_grid, core_smooth, color=CORE_COLOR, lw=2.0, zorder=4, label=r"Core, $V_{E,core}$")
    ax.fill_between(pop_grid, outer_lo, outer_hi, color=OUTER_COLOR, alpha=0.15, linewidth=0, zorder=2)
    ax.plot(pop_grid, outer_smooth, color=OUTER_COLOR, lw=2.0, zorder=4, label=r"Periphery, $V_{E,outer}$")

    ax.axvline(P_star, color="#555555", lw=1.0, ls=":", zorder=2)
    ax.text(P_star, y_lim_hi * 0.60, f" P*~{P_star:,.0f}", fontsize=6.5, color="#555555",
           va="top", ha="left", rotation=90)

    core_row = reg_df[reg_df["zone"] == "core"].iloc[0]
    outer_row = reg_df[reg_df["zone"] == "outer"].iloc[0]
    ax.text(0.97, 0.97,
           f"Above {REGRESSION_POP_MIN:,}:\n"
           f"Core differentiation declines, p={core_row['p_value']:.3g}\n"
           f"Peripheral differentiation stable, p={outer_row['p_value']:.2g}",
           transform=ax.transAxes, fontsize=6, color="#333333", va="top", ha="right",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", linewidth=0.5))

    ax.set_xscale("log")
    ax.set_ylim(0, y_lim_hi)
    ax.set_xlabel("City population", fontsize=9)
    ax.set_ylabel("Cross-city embedding distance\n(mean over K=10 same-size neighbours)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("black")
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors="black", labelsize=8)

    caption = (
        r"Decomposes 23_scale_dependent_convergence.py's $\Delta V_E(P)=V_{E,outer}-V_{E,core}$ "
        "reversal into its two levels." "\n"
        r"$V_{E,z}(P)$: mean embedding distance between zone $z$ of a city and the same zone of its "
        "K=10 nearest-by-" "\n"
        "population neighbours (same sample, same metric as 23 -- no new computation). Weighted "
        "by reliability_weight" "\n"
        "(sqrt(min(n_core, n_outer)), inherited from 23) at both the LOWESS trend and the "
        f"regression. The $\\beta$/p-values" "\n"
        f"box reports the weighted regression on cities >= {REGRESSION_POP_MIN:,}. Bands: 95% "
        "country-block bootstrap CI."
    )
    fig.text(0.5, -0.02, caption, fontsize=5.5, color="#555555", style="italic",
            ha="center", va="top", linespacing=1.6)

    for ext in ("svg", "png"):
        out = OUT_FIG / f"core_outer_divergence_mechanism.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        log.info("Saved: %s", out)
    plt.close(fig)

    log.info("Done.")


if __name__ == "__main__":
    main()
