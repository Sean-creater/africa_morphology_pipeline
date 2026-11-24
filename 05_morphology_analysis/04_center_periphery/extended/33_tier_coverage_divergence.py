"""
33_tier_coverage_divergence.py
=================================
Answers a different question from 30/32: not "how much does coverage
explain the SMALL-vs-LARGE composition gap" (32's coverage-adjustment),
but, WITHIN each size tier separately, "at a given built-up coverage
level, how different do same-tier cities look from each other" -- and
does that answer depend on core vs periphery.

    V_between(tier, zone, c) = mean_{c' in tier, c' != c}
                               [1 - cos(mean_emb(c, zone), mean_emb(c', zone))]

evaluated at city c's own mean coverage in that zone, then a LOWESS
trend of V_between against coverage is fit per tier, separately for core
and periphery (two figures, not one -- core/outer zones, qualifying
city sample, and core_emb/outer_emb construction all reused byte-for-
byte from 23_scale_dependent_convergence.py's scale_dependent_
dispersion.csv, not recomputed).

This is a full within-tier pairwise comparison (not 23's K=10 nearest-
by-population neighbours) -- population plays no role here at all
beyond having already sorted cities into the 3 Africapolis tiers; within
a tier, every city is compared to every other city in that SAME tier,
at ITS OWN coverage level. Answers: "at 50% built-up cover, how
unpredictable is a small city's core, vs a large city's core" --
directly extending the SDG 11.3 "coverage alone is an incomplete
description of urban form" argument to each size tier individually,
rather than comparing tiers to each other (that is 29's and 32's job).

Method
------
- Same qualifying sample as 23 (core q<0.2 AND outer q>0.75, >=1 cell
  each zone -- methods review, 2026-10-05: the previous >=20 cells/zone
  cutoff excluded the same systematically-small-city-biased ~75% of
  cities documented in 23's docstring; now every city with a non-empty
  core and outer qualifies, with a reliability weight sqrt(min(n_core,
  n_outer)) applied at the LOWESS/bootstrap aggregation step instead of
  an inclusion filter) -- core_emb / outer_emb reused directly from 23's
  own per-city mean-embedding construction (re-derived here from
  cell_embeddings_index.parquet with the identical zone definitions, not
  read from 23's output file, since 23 does not save the raw per-city
  embedding vectors, only the already-aggregated K=10 dispersion table).
- Per city, per zone: mean LCF_Built across that zone's cells (x-axis
  value) and V_between against every OTHER city in the SAME tier (not
  the whole city pool) -- a full pairwise comparison, since tier sizes
  make O(n^2) trivial, unlike 23's continental pool which needed a
  K-nearest-neighbour design to avoid O(n^2) there.
- LOWESS trend of V_between vs coverage, per tier, WEIGHTED by each
  city's reliability_weight (integer row-repetition, same convention as
  23), country-block bootstrap 95% CI (resample countries, each
  contributing its full within-tier city set, preserving multiplicity --
  this project's standard design).
- Scatter style: point size now scales with reliability_weight (same
  convention as 23/27/29), alpha=0.18 as before.
- Tier colours: TIER_COLORS, exact values from
  20_trajectory_by_africapolis_tier.py / 25_trajectory_by_tier_
  combined.py (canonical, not re-derived).

Output
------
  outputs/tables/tier_coverage_divergence_core.csv
      tier | city_key | coverage | V_between
  outputs/tables/tier_coverage_divergence_outer.csv
      (same columns, outer zone)
  outputs/tables/tier_coverage_divergence_trend.csv
      zone | tier | coverage_grid | V_between_smooth | V_between_lo | V_between_hi
  outputs/figures/tier_coverage_divergence_core.svg/.png
  outputs/figures/tier_coverage_divergence_outer.svg/.png

Usage
-----
    python 33_tier_coverage_divergence.py
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
OUT_ARRAYS = ROOT / "outputs" / "arrays"
INDEX_PQ   = OUT_TABLES / "cell_embeddings_index.parquet"
EMB_NPY    = OUT_ARRAYS / "cell_embeddings.npy"
BIN_GRID   = OUT_TABLES / "bin_grid.json"
OUT_FIG    = ROOT / "outputs" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

CORE_Q_MAX     = 0.2
OUTER_Q_MIN    = 0.75
MIN_CELLS_ZONE = 1      # inclusion floor (methods review: was 20); see
                        # module docstring -- reliability_weight handles
                        # small-n cities at aggregation instead
LOWESS_FRAC    = 0.6
N_BOOT         = 500
BOOT_SEED      = 42
GRID_N         = 60

TIER_EDGES  = [0, 100_000, 1_000_000, np.inf]
TIER_LABELS = ["Small (<100k)", "Intermediate (100k-1M)", "Large (>=1M)"]
TIER_COLORS = {"Small (<100k)": "#8DA0CB", "Intermediate (100k-1M)": "#FFD700",
              "Large (>=1M)": "#C8292A"}


def _cos_dist_to_set(v: np.ndarray, V: np.ndarray) -> np.ndarray:
    nv = np.linalg.norm(v)
    nV = np.linalg.norm(V, axis=1)
    nV[nV < 1e-12] = np.nan
    if nv < 1e-12:
        return np.full(len(V), np.nan)
    sims = (V @ v) / (nV * nv)
    return 1.0 - sims


def _lowess_trend(x: np.ndarray, y: np.ndarray, grid: np.ndarray,
                  frac: float = LOWESS_FRAC, weight: np.ndarray = None) -> np.ndarray:
    """Weighted via integer row-repetition, same convention as
    23_scale_dependent_convergence.py's _lowess_trend."""
    mask = np.isfinite(x) & np.isfinite(y)
    if weight is not None:
        mask &= np.isfinite(weight) & (weight > 0)
    if mask.sum() < 10:
        return np.full(len(grid), np.nan)
    xx, yy = x[mask], y[mask]
    if weight is not None:
        reps = np.maximum(np.round(weight[mask]).astype(int), 1)
        xx = np.repeat(xx, reps)
        yy = np.repeat(yy, reps)
    sm = lowess(yy, xx, frac=frac, return_sorted=True)
    return np.interp(grid, sm[:, 0], sm[:, 1])


def _within_tier_v_between(tier_keys: list, emb_map: dict) -> dict:
    """city_key -> mean cosine distance to every OTHER city in tier_keys."""
    mat = np.array([emb_map[ck] for ck in tier_keys])
    out = {}
    for i, ck in enumerate(tier_keys):
        others = np.delete(mat, i, axis=0)
        out[ck] = float(np.nanmean(_cos_dist_to_set(mat[i], others)))
    return out


def main():
    log.info("Loading cell_embeddings_index.parquet + cell_embeddings.npy ...")
    idx = pd.read_parquet(INDEX_PQ,
                          columns=["city", "country", "row_idx", "q_relative",
                                   "log_pop", "LCF_Built"])
    emb = np.load(EMB_NPY)
    with open(BIN_GRID) as f:
        q_max = float(json.load(f)["q_P95"])
    idx = idx[(idx["q_relative"] >= 0) & (idx["q_relative"] <= q_max)].copy()
    idx["_city_key"] = idx["country"] + "||" + idx["city"]
    idx = idx.dropna(subset=["log_pop", "LCF_Built"])
    log.info("  %d cells, q_max=%.4f", len(idx), q_max)

    # ── same qualifying-city sample as 23 ──
    core_sub = idx[idx["q_relative"] < CORE_Q_MAX]
    outer_sub = idx[(idx["q_relative"] > OUTER_Q_MIN) & (idx["q_relative"] <= q_max)]
    core_n = core_sub.groupby("_city_key").size()
    outer_n = outer_sub.groupby("_city_key").size()
    qualified = sorted(set(core_n[core_n >= MIN_CELLS_ZONE].index) &
                       set(outer_n[outer_n >= MIN_CELLS_ZONE].index))
    log.info("  %d cities qualify (>=%d cells in BOTH core and outer).",
            len(qualified), MIN_CELLS_ZONE)

    city_log_pop = idx.groupby("_city_key")["log_pop"].first()
    city_country = {ck: ck.split("||")[0] for ck in qualified}
    city_pop = np.exp(city_log_pop[qualified])
    tier_map = pd.cut(city_pop, bins=TIER_EDGES, labels=TIER_LABELS, right=False)
    city_weight = {ck: float(np.sqrt(min(core_n[ck], outer_n[ck]))) for ck in qualified}
    log.info("  reliability weight: median=%.1f, min=%.1f, max=%.1f",
            np.median(list(city_weight.values())), min(city_weight.values()), max(city_weight.values()))

    core_emb, outer_emb, core_cov, outer_cov = {}, {}, {}, {}
    for ck in qualified:
        c_rows = core_sub[core_sub["_city_key"] == ck]
        o_rows = outer_sub[outer_sub["_city_key"] == ck]
        core_emb[ck] = emb[c_rows["row_idx"].values].mean(axis=0)
        outer_emb[ck] = emb[o_rows["row_idx"].values].mean(axis=0)
        core_cov[ck] = float(c_rows["LCF_Built"].mean())
        outer_cov[ck] = float(o_rows["LCF_Built"].mean())

    cities_by_tier = {t: [ck for ck in qualified if tier_map[ck] == t] for t in TIER_LABELS}
    for t in TIER_LABELS:
        log.info("  %s: %d cities", t, len(cities_by_tier[t]))

    for zone, emb_map, cov_map in [("core", core_emb, core_cov), ("outer", outer_emb, outer_cov)]:
        log.info("=== Zone: %s ===", zone)
        rows = []
        for tier in TIER_LABELS:
            tier_keys = cities_by_tier[tier]
            v_between = _within_tier_v_between(tier_keys, emb_map)
            for ck in tier_keys:
                rows.append({"tier": tier, "city_key": ck, "country": city_country[ck],
                            "coverage": cov_map[ck], "V_between": v_between[ck],
                            "reliability_weight": city_weight[ck]})
        main_df = pd.DataFrame(rows)
        main_df.to_csv(OUT_TABLES / f"tier_coverage_divergence_{zone}.csv",
                       index=False, float_format="%.6f")
        log.info("  Saved: tier_coverage_divergence_%s.csv (%d rows)", zone, len(main_df))

        # ── LOWESS trend + country-block bootstrap, per tier ──
        cov_min = main_df["coverage"].min()
        cov_max = main_df["coverage"].max()
        cov_grid = np.linspace(cov_min, cov_max, GRID_N)
        rng = np.random.default_rng(BOOT_SEED)

        trend_rows = []
        for tier in TIER_LABELS:
            sub = main_df[main_df["tier"] == tier]
            smooth = _lowess_trend(sub["coverage"].values, sub["V_between"].values, cov_grid,
                                   weight=sub["reliability_weight"].values)

            countries = sorted(sub["country"].unique())
            country_df = {c: sub[sub["country"] == c] for c in countries}
            boot = np.full((N_BOOT, GRID_N), np.nan)
            for b in range(N_BOOT):
                chosen = rng.choice(countries, size=len(countries), replace=True)
                resampled = pd.concat([country_df[c] for c in chosen], ignore_index=True)
                if len(resampled) < 10:
                    continue
                boot[b] = _lowess_trend(resampled["coverage"].values, resampled["V_between"].values, cov_grid,
                                        weight=resampled["reliability_weight"].values)
            lo = np.nanpercentile(boot, 2.5, axis=0)
            hi = np.nanpercentile(boot, 97.5, axis=0)

            for i in range(GRID_N):
                trend_rows.append({"tier": tier, "coverage_grid": cov_grid[i],
                                  "V_between_smooth": smooth[i], "V_between_lo": lo[i], "V_between_hi": hi[i]})
            log.info("  %s: n=%d  coverage range [%.1f, %.1f]  V_between range [%.3f, %.3f]",
                    tier, len(sub), sub["coverage"].min(), sub["coverage"].max(),
                    sub["V_between"].min(), sub["V_between"].max())

        trend_df = pd.DataFrame(trend_rows)
        trend_df.to_csv(OUT_TABLES / f"tier_coverage_divergence_trend_{zone}.csv",
                        index=False, float_format="%.6f")
        log.info("  Saved: tier_coverage_divergence_trend_%s.csv", zone)

        # ── figure ──
        log.info("  Drawing figure ...")
        AXES_W_MM, AXES_H_MM = 100.0, 75.0
        LEFT_MM, RIGHT_MM, BOTTOM_MM, TOP_MM = 20.0, 20.0, 16.0, 6.0
        FIG_W_MM = LEFT_MM + AXES_W_MM + RIGHT_MM
        FIG_H_MM = BOTTOM_MM + AXES_H_MM + TOP_MM
        fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
        ax = fig.add_axes([LEFT_MM / FIG_W_MM, BOTTOM_MM / FIG_H_MM,
                           AXES_W_MM / FIG_W_MM, AXES_H_MM / FIG_H_MM])

        for tier in TIER_LABELS:
            color = TIER_COLORS[tier]
            sub = main_df[main_df["tier"] == tier]
            point_size = 2 + 8 * (sub["reliability_weight"] / main_df["reliability_weight"].max())
            ax.scatter(sub["coverage"], sub["V_between"], s=point_size, alpha=0.18,
                      color=color, linewidth=0, zorder=1)
            tsub = trend_df[trend_df["tier"] == tier].sort_values("coverage_grid")
            ax.fill_between(tsub["coverage_grid"], tsub["V_between_lo"], tsub["V_between_hi"],
                            color=color, alpha=0.15, linewidth=0, zorder=2)
            ax.plot(tsub["coverage_grid"], tsub["V_between_smooth"], color=color, lw=2.0,
                   zorder=3, label=f"{tier.split(' (')[0]} (n={main_df[main_df['tier']==tier]['city_key'].nunique()})")

        ax.set_xlabel("Built-up coverage, LCF_Built (%)", fontsize=9)
        zone_label = "core (q<0.2)" if zone == "core" else "periphery (q>0.75)"
        ax.set_ylabel(f"Cross-city divergence within tier\n($V_{{between}}$, {zone_label})", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="upper right" if zone == "core" else "lower right")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color("black")
            ax.spines[sp].set_linewidth(0.8)
        ax.tick_params(colors="black", labelsize=8)

        caption = (
            f"Within each Africapolis size tier, full pairwise cross-city embedding distance "
            f"($V_{{between}}$) at the" "\n"
            f"{zone_label}, evaluated at each city's own mean built-up coverage -- "
            "NOT a tier-vs-tier comparison" "\n"
            "(that is 29/32's job); this asks, within one size tier, how unpredictable same-"
            "coverage cities are" "\n"
            "from each other. Points: cities, sized by reliability_weight (n shown in legend). "
            "Curves: weighted" "\n"
            "LOWESS trend per tier. Bands: 95% country-block bootstrap CI. Same qualifying "
            "sample and core/outer" "\n"
            "zone definitions as 23_scale_dependent_convergence.py. No population density or "
            "Pop2022 used --" "\n"
            "population only defines the size tiers; LCF_Built is remote-sensing coverage."
        )
        fig.text(0.5, -0.02, caption, fontsize=5.5, color="#555555", style="italic",
                ha="center", va="top", linespacing=1.6)

        for ext in ("svg", "png"):
            out = OUT_FIG / f"tier_coverage_divergence_{zone}.{ext}"
            fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
            log.info("  Saved: %s", out)
        plt.close(fig)

    log.info("Done.")


if __name__ == "__main__":
    main()
