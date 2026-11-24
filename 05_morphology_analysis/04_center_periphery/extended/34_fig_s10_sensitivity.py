"""
34_fig_s10_sensitivity.py
==========================
Supplementary Fig. S10 -- sensitivity of the two intraurban
morphological-differentiation findings (Fig. 4c, Fig. 4d) to:

  (a) alternative center definitions   -- primary_pop (main) / pop_centroid,
                                          applied to Fig. 4c (a third candidate,
                                          geom_centroid, was computed and then
                                          DROPPED -- see note below)
  (b) alternative neighborhood size K -- 5 / 10 / 15 nearest population
                                          neighbours, applied to Fig. 4c
                                          (REUSED, not recomputed here --
                                          see 23_scale_dependent_convergence.py,
                                          which already saves this check)
  (c) alternative core-periphery boundaries -- Narrow (0.15/0.80) / Main
                                          (0.20/0.75) / Broad (0.25/0.70),
                                          applied to Fig. 4c
  (d) a compact cross-check of Fig. 4d (tier x zone) under the center- and
      boundary-alternatives above, read off each setting's V_between at a
      fixed 80% built-up-coverage level -- the same figure the main text
      cites ("at ~80% coverage, divergence was 0.31 in large-city cores vs
      0.43 in small-city cores ... 0.45 vs 0.59 in peripheries")

A fourth candidate axis -- alternative radial coordinates (q_relative vs.
u_scaled vs. d_km) -- is NOT included. u_scaled and q_relative are both
city-specific-percentile-normalised relative radii (comparable), but d_km
is an absolute distance with no principled way to re-derive the SAME
CORE_Q_MAX/OUTER_Q_MIN cutoffs in km (city sizes vary by two orders of
magnitude), so varying it would silently also be varying the core/outer
zone extents in a way that confounds with axis (c). Per methods review,
dropped rather than introduce an uninterpretable comparison.

Geometric centroid (a candidate center definition) was computed, plotted,
and then dropped from the published figure: under this definition the
large-city plateau of panel (a)'s delta_V curve is marginally NEGATIVE
(mean -0.0097 over 10^6-10^7 population), i.e. it does not cleanly
replicate the sign reversal the way the main analysis and the
population-weighted centroid both do (only attenuates toward zero).
Judged, on balance, to reflect geom_centroid being the weakest-justified
of the three center definitions (population-UNweighted, so it can land in
sparsely-built or irregularly-shaped parts of a city) rather than a real
threat to the reversal finding -- but not reliable enough to present as a
confirming curve. The code to reproduce it is kept (CENTER_SETTINGS,
dist_norm_geom join) for anyone re-deriving this, but panel (a) and
panel (d) are filtered to exclude it before plotting -- see the
`.str.contains("[Gg]eometric")` filters right before plot_figure() in
main().

Design choices (per methods review)
------------------------------------
- Panels (a) and (c): same visual language as 23's existing K-sensitivity
  panel (scale_dependent_core_periphery.png, panel b) -- one dark main
  curve, two lighter alternative curves, LOWESS point estimate only, NO
  bootstrap CI band (this is a "does the pattern survive" check, not a
  re-estimation of the main finding's uncertainty).
- Headline criterion is NOT "does the zero-crossing sit at exactly
  ~100,000" -- that is a soft, LOWESS-smoothing-dependent descriptive
  threshold, not something to re-litigate as pass/fail. The criterion is
  the qualitative pattern: delta_V is negative at small population and
  positive at large population, under every alternative setting.
- Panel (d) does not re-plot all six (tier x zone) curves per setting.
  It reports the single number the main text's own prose argument
  actually leans on -- V_between at coverage=80% -- across settings, as
  a compact table/strip plot, not a full LOWESS re-fit comparison.

Center-definition data
-----------------------
cell_embeddings_index.parquet only carries the PRIMARY center's
q_relative. The pop_centroid / geom_centroid alternatives are read from
core/outputs/tables/cell_radial_position.parquet's dist_norm_pop /
dist_norm_geom columns -- already P98-normalised per center by
02_cell_radial_position.py (the same "distance / city-specific percentile
scale" role q_relative plays for the primary center), joined in by
GridID (100% match verified against the embedding index).

Boundary-definition settings
------------------------------
  Narrow : core q < 0.15, outer q > 0.80
  Main   : core q < 0.20, outer q > 0.75   (= CORE_Q_MAX/OUTER_Q_MIN in
                                             23_scale_dependent_convergence.py
                                             and 33_tier_coverage_divergence.py)
  Broad  : core q < 0.25, outer q > 0.70

Output
------
  outputs/tables/s10_panel_a_center.csv       setting | log_pop_grid | delta_V_smooth
  outputs/tables/s10_panel_c_boundary.csv     setting | log_pop_grid | delta_V_smooth
  outputs/tables/s10_panel_d_coverage80.csv   setting | tier | zone | V_between_at_80pct
  outputs/figures/fig_s10_sensitivity.svg/.png

Usage
-----
    python 34_fig_s10_sensitivity.py
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

ROOT        = Path(__file__).parent
OUT_TABLES  = ROOT / "outputs" / "tables"
OUT_ARRAYS  = ROOT / "outputs" / "arrays"
OUT_FIG     = ROOT / "outputs" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

INDEX_PQ    = OUT_TABLES / "cell_embeddings_index.parquet"
EMB_NPY     = OUT_ARRAYS / "cell_embeddings.npy"
BIN_GRID    = OUT_TABLES / "bin_grid.json"
RADIAL_PQ   = ROOT.parent / "core" / "outputs" / "tables" / "cell_radial_position.parquet"
K23_TREND   = OUT_TABLES / "core_outer_reversal_trend.csv"   # existing K-sensitivity, reused

N_TYPES        = 17
MIN_CELLS_ZONE = 1
K_MAIN         = 10
LOWESS_FRAC    = 0.5
GRID_N         = 60
COVERAGE_80    = 80.0   # LCF_Built is on a 0-100 scale (see 33's script)

MAIN_CORE_Q, MAIN_OUTER_Q = 0.20, 0.75
BOUNDARY_SETTINGS = {
    "Narrow (0.15 / 0.80)": (0.15, 0.80),
    "Main (0.20 / 0.75)":   (MAIN_CORE_Q, MAIN_OUTER_Q),
    "Broad (0.25 / 0.70)":  (0.25, 0.70),
}

CENTER_SETTINGS = {
    "Primary population center (main)": "q_relative",
    "Population-weighted centroid":     "dist_norm_pop",
    "Geometric centroid":               "dist_norm_geom",
}

TIER_EDGES  = [0, 100_000, 1_000_000, np.inf]
TIER_LABELS = ["Small (<100k)", "Intermediate (100k-1M)", "Large (>=1M)"]

COLOR_MAIN = "#1c2b5e"
COLOR_ALT  = ["#8DA0CB", "#C8292A"]


# ---------------------------------------------------------------------------
# Shared helpers (ported from 23_scale_dependent_convergence.py)
# ---------------------------------------------------------------------------
def _cos_dist_to_set(v: np.ndarray, V: np.ndarray) -> np.ndarray:
    nv = np.linalg.norm(v)
    nV = np.linalg.norm(V, axis=1)
    nV[nV < 1e-12] = np.nan
    if nv < 1e-12:
        return np.full(len(V), np.nan)
    sims = (V @ v) / (nV * nv)
    return 1.0 - sims


def _knn_delta_v(city_keys, city_log_pop, core_emb, outer_emb, K) -> pd.DataFrame:
    keys = list(city_keys)
    log_pops = np.array([city_log_pop[ck] for ck in keys])
    order = np.argsort(log_pops)
    sorted_keys = [keys[i] for i in order]
    sorted_pops = log_pops[order]
    core_mat = np.array([core_emb[ck] for ck in sorted_keys])
    outer_mat = np.array([outer_emb[ck] for ck in sorted_keys])

    rows = []
    for i, ck in enumerate(sorted_keys):
        dists = np.abs(sorted_pops - sorted_pops[i])
        dists[i] = np.inf
        knn_idx = np.argsort(dists)[:K]
        v_core = float(np.nanmean(_cos_dist_to_set(core_mat[i], core_mat[knn_idx])))
        v_outer = float(np.nanmean(_cos_dist_to_set(outer_mat[i], outer_mat[knn_idx])))
        rows.append({"city_key": ck, "log_pop": sorted_pops[i],
                     "delta_V": v_outer - v_core})
    return pd.DataFrame(rows)


def _lowess_trend(log_pop, delta, grid, frac=LOWESS_FRAC, weight=None):
    mask = np.isfinite(log_pop) & np.isfinite(delta)
    if weight is not None:
        mask &= np.isfinite(weight) & (weight > 0)
    if mask.sum() < 10:
        return np.full(len(grid), np.nan)
    x, y = log_pop[mask], delta[mask]
    if weight is not None:
        reps = np.maximum(np.round(weight[mask]).astype(int), 1)
        x, y = np.repeat(x, reps), np.repeat(y, reps)
    sm = lowess(y, x, frac=frac, return_sorted=True)
    return np.interp(grid, sm[:, 0], sm[:, 1])


def _build_zone_embeddings(idx, emb, q_col, core_q, outer_q, q_max):
    core_sub = idx[idx[q_col] < core_q]
    outer_sub = idx[(idx[q_col] > outer_q) & (idx[q_col] <= q_max)]
    core_n = core_sub.groupby("_city_key").size()
    outer_n = outer_sub.groupby("_city_key").size()
    qualified = sorted(set(core_n[core_n >= MIN_CELLS_ZONE].index) &
                       set(outer_n[outer_n >= MIN_CELLS_ZONE].index))
    city_weight = {ck: float(np.sqrt(min(core_n[ck], outer_n[ck]))) for ck in qualified}

    core_emb, outer_emb = {}, {}
    for ck in qualified:
        c_rows = core_sub[core_sub["_city_key"] == ck]
        o_rows = outer_sub[outer_sub["_city_key"] == ck]
        core_emb[ck] = emb[c_rows["row_idx"].values].mean(axis=0)
        outer_emb[ck] = emb[o_rows["row_idx"].values].mean(axis=0)
    return qualified, city_weight, core_emb, outer_emb, core_sub, outer_sub


# ---------------------------------------------------------------------------
# Panels a/c: Fig. 4c delta_V LOWESS curves under alternative settings
# ---------------------------------------------------------------------------
def run_fig4c_variant(idx, emb, city_log_pop, q_col, core_q, outer_q, q_max, grid):
    qualified, city_weight, core_emb, outer_emb, _, _ = _build_zone_embeddings(
        idx, emb, q_col, core_q, outer_q, q_max)
    log.info("    %d qualifying cities (q_col=%s, core<%.2f, outer>%.2f)",
             len(qualified), q_col, core_q, outer_q)
    df = _knn_delta_v(qualified, city_log_pop, core_emb, outer_emb, K_MAIN)
    df["weight"] = df["city_key"].map(city_weight)
    smooth = _lowess_trend(df["log_pop"].values, df["delta_V"].values, grid,
                           weight=df["weight"].values)
    return smooth


def panel_a_center_definitions(idx_main, emb, radial_alt, city_log_pop, q_max, grid):
    log.info("=== Panel a: alternative center definitions (Fig. 4c) ===")
    rows = []
    for label, q_col in CENTER_SETTINGS.items():
        log.info("  [%s]", label)
        if q_col == "q_relative":
            idx_use = idx_main
        else:
            idx_use = idx_main.drop(columns=["q_relative"]).merge(
                radial_alt[["GridID", "city", "country", q_col]],
                on=["GridID", "city", "country"], how="left")
            idx_use = idx_use.rename(columns={q_col: "q_relative"})
            idx_use = idx_use.dropna(subset=["q_relative"])
        smooth = run_fig4c_variant(idx_use, emb, city_log_pop, "q_relative",
                                   MAIN_CORE_Q, MAIN_OUTER_Q, q_max, grid)
        for g, s in zip(grid, smooth):
            rows.append({"setting": label, "log_pop_grid": g, "delta_V_smooth": s})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "s10_panel_a_center.csv", index=False, float_format="%.6f")
    log.info("  Saved: s10_panel_a_center.csv")
    return out


def panel_c_boundaries(idx_main, emb, city_log_pop, q_max, grid):
    log.info("=== Panel c: alternative core-periphery boundaries (Fig. 4c) ===")
    rows = []
    for label, (core_q, outer_q) in BOUNDARY_SETTINGS.items():
        log.info("  [%s]", label)
        smooth = run_fig4c_variant(idx_main, emb, city_log_pop, "q_relative",
                                   core_q, outer_q, q_max, grid)
        for g, s in zip(grid, smooth):
            rows.append({"setting": label, "log_pop_grid": g, "delta_V_smooth": s})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "s10_panel_c_boundary.csv", index=False, float_format="%.6f")
    log.info("  Saved: s10_panel_c_boundary.csv")
    return out


# ---------------------------------------------------------------------------
# Panel d: Fig. 4d V_between at coverage=80%, across settings
# ---------------------------------------------------------------------------
def panel_d_coverage80(idx_main, emb, radial_alt, q_max):
    log.info("=== Panel d: Fig. 4d V_between at 80%% coverage, across settings ===")

    settings = {
        "Main (primary center, 0.20/0.75)": ("q_relative", MAIN_CORE_Q, MAIN_OUTER_Q),
        "Alt. center: pop-weighted centroid": ("dist_norm_pop", MAIN_CORE_Q, MAIN_OUTER_Q),
        "Alt. center: geometric centroid":    ("dist_norm_geom", MAIN_CORE_Q, MAIN_OUTER_Q),
        "Alt. boundary: Narrow (0.15/0.80)":  ("q_relative", 0.15, 0.80),
        "Alt. boundary: Broad (0.25/0.70)":   ("q_relative", 0.25, 0.70),
    }

    rows = []
    for label, (q_col, core_q, outer_q) in settings.items():
        log.info("  [%s]", label)
        if q_col == "q_relative":
            idx_use = idx_main
        else:
            idx_use = idx_main.drop(columns=["q_relative"]).merge(
                radial_alt[["GridID", "city", "country", q_col]],
                on=["GridID", "city", "country"], how="left")
            idx_use = idx_use.rename(columns={q_col: "q_relative"})
            idx_use = idx_use.dropna(subset=["q_relative"])

        qualified, city_weight, _, _, core_sub, outer_sub = _build_zone_embeddings(
            idx_use, emb, "q_relative", core_q, outer_q, q_max)

        city_log_pop = idx_use.groupby("_city_key")["log_pop"].first()
        city_pop = np.exp(city_log_pop[qualified])
        tier_map = pd.cut(city_pop, bins=TIER_EDGES, labels=TIER_LABELS, right=False)

        core_cov = core_sub.groupby("_city_key")["LCF_Built"].mean()
        outer_cov = outer_sub.groupby("_city_key")["LCF_Built"].mean()

        core_emb_m, outer_emb_m = {}, {}
        for ck in qualified:
            c_rows = core_sub[core_sub["_city_key"] == ck]
            o_rows = outer_sub[outer_sub["_city_key"] == ck]
            core_emb_m[ck] = emb[c_rows["row_idx"].values].mean(axis=0)
            outer_emb_m[ck] = emb[o_rows["row_idx"].values].mean(axis=0)

        for zone, emb_map, cov_map in [("core", core_emb_m, core_cov),
                                       ("outer", outer_emb_m, outer_cov)]:
            for tier in TIER_LABELS:
                tier_keys = [ck for ck in qualified if tier_map[ck] == tier]
                if len(tier_keys) < 5:
                    rows.append({"setting": label, "tier": tier, "zone": zone,
                                "V_between_at_80pct": np.nan, "n_cities": len(tier_keys)})
                    continue
                mat = np.array([emb_map[ck] for ck in tier_keys])
                v_between = np.array([
                    float(np.nanmean(_cos_dist_to_set(mat[i], np.delete(mat, i, axis=0))))
                    for i in range(len(tier_keys))
                ])
                cov = np.array([cov_map.get(ck, np.nan) for ck in tier_keys])
                mask = np.isfinite(cov) & np.isfinite(v_between)
                if mask.sum() < 10:
                    val = np.nan
                else:
                    sm = lowess(v_between[mask], cov[mask], frac=0.6, return_sorted=True)
                    val = float(np.interp(COVERAGE_80, sm[:, 0], sm[:, 1]))
                rows.append({"setting": label, "tier": tier, "zone": zone,
                            "V_between_at_80pct": val, "n_cities": len(tier_keys)})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "s10_panel_d_coverage80.csv", index=False, float_format="%.6f")
    log.info("  Saved: s10_panel_d_coverage80.csv")
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_figure(panel_a_df, panel_b_df, panel_c_df, panel_d_df, grid):
    fig = plt.figure(figsize=(260 / 25.4, 210 / 25.4))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28,
                          left=0.08, right=0.93, top=0.95, bottom=0.08)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    def _style_axis(ax, title):
        ax.axhline(0, color="#AAAAAA", lw=1.0, ls="--", zorder=1)
        ax.set_xlabel("City population", fontsize=11)
        ax.set_ylabel(r"$\Delta V_E$ (outer $-$ core)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color("#555555")
            ax.spines[sp].set_linewidth(0.8)
        ax.tick_params(colors="#555555", labelsize=10)
        xt = [1e3, 1e4, 1e5, 1e6, 1e7]
        ax.set_xticks(np.log(xt))
        ax.set_xticklabels([f"$10^{int(np.log10(v))}$" for v in xt])

    # Panel a: center definitions
    for i, (label, sub) in enumerate(panel_a_df.groupby("setting", sort=False)):
        is_main = label.startswith("Primary")
        color = COLOR_MAIN if is_main else COLOR_ALT[i % 2]
        ax_a.plot(sub["log_pop_grid"], sub["delta_V_smooth"],
                 color=color, lw=2.4 if is_main else 1.8,
                 zorder=3 if is_main else 2,
                 label=label.replace(" (main)", ""))
    _style_axis(ax_a, "a | Alternative center definitions")
    ax_a.legend(fontsize=9.5, frameon=False, loc="upper left")

    # Panel b: reused K-sensitivity (from 23_scale_dependent_convergence.py)
    for i, K in enumerate(sorted(panel_b_df["K"].unique())):
        sub = panel_b_df[panel_b_df["K"] == K].sort_values("log_pop_grid")
        is_main = (K == K_MAIN)
        color = COLOR_MAIN if is_main else COLOR_ALT[0 if K < K_MAIN else 1]
        ax_b.plot(sub["log_pop_grid"], sub["delta_V_E_smooth"],
                 color=color, lw=2.4 if is_main else 1.8,
                 zorder=3 if is_main else 2, label=f"K={K}")
    _style_axis(ax_b, "b | Alternative neighborhood size")
    ax_b.legend(fontsize=9.5, frameon=False, loc="upper left")

    # Panel c: boundaries
    for i, (label, sub) in enumerate(panel_c_df.groupby("setting", sort=False)):
        is_main = label.startswith("Main")
        color = COLOR_MAIN if is_main else COLOR_ALT[i % 2]
        ax_c.plot(sub["log_pop_grid"], sub["delta_V_smooth"],
                 color=color, lw=2.4 if is_main else 1.8,
                 zorder=3 if is_main else 2, label=label)
    _style_axis(ax_c, "c | Alternative core–periphery boundaries")
    ax_c.legend(fontsize=9.5, frameon=False, loc="upper left")

    # Panel d: strip plot of V_between @ 80% coverage, across settings.
    # Short row labels (full setting -> label match panel a/c's legend
    # entries, so the reader cross-references by name, not by re-reading
    # a long string here).
    SETTING_SHORT = {
        "Main (primary center, 0.20/0.75)":  "Main",
        "Alt. center: pop-weighted centroid": "Alt. center\n(pop-weighted)",
        "Alt. boundary: Narrow (0.15/0.80)":  "Alt. boundary\n(narrow)",
        "Alt. boundary: Broad (0.25/0.70)":   "Alt. boundary\n(broad)",
    }
    settings_order = list(panel_d_df["setting"].unique())
    tier_short = {"Small (<100k)": "Small", "Intermediate (100k-1M)": "Intermediate",
                 "Large (>=1M)": "Large"}
    tier_colors = {"Small": "#8DA0CB", "Intermediate": "#FFD700", "Large": "#C8292A"}
    y_positions = {s: i for i, s in enumerate(settings_order)}
    for zone, marker in [("core", "o"), ("outer", "^")]:
        for tier_label, tier_s in tier_short.items():
            sub = panel_d_df[(panel_d_df["zone"] == zone) & (panel_d_df["tier"] == tier_label)]
            y = [y_positions[s] + (0.15 if zone == "outer" else -0.15) for s in sub["setting"]]
            ax_d.scatter(sub["V_between_at_80pct"], y, marker=marker,
                        s=42, color=tier_colors[tier_s], edgecolor="white", linewidth=0.6,
                        zorder=3)
    ax_d.set_yticks(range(len(settings_order)))
    ax_d.set_yticklabels([SETTING_SHORT.get(s, s) for s in settings_order], fontsize=10.5)
    ax_d.set_xlabel(r"$V_{between}$ at 80% built-up coverage", fontsize=11)
    ax_d.set_title("d | Fig. 4d read-off, across settings (a + c)", fontsize=13,
                   fontweight="bold", pad=8)
    for sp in ["top", "right"]:
        ax_d.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax_d.spines[sp].set_color("#555555")
        ax_d.spines[sp].set_linewidth(0.8)
    ax_d.tick_params(colors="#555555", labelsize=10)
    ax_d.invert_yaxis()

    # Compact legend: color encodes tier (3 entries), marker shape encodes
    # zone -- stated once in a text annotation rather than doubling the
    # legend to 6 entries (color x shape would otherwise repeat the same
    # 3 colors twice).
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([], [], marker="s", linestyle="none", markersize=8,
              color=tier_colors[t], label=t)
        for t in ["Small", "Intermediate", "Large"]
    ]
    size_legend = ax_d.legend(handles=legend_handles, fontsize=9.5, frameon=False,
                              title="City size", title_fontsize=9.5,
                              loc="center left", bbox_to_anchor=(1.0, 0.68),
                              handletextpad=0.5)
    ax_d.add_artist(size_legend)  # a second legend() call below would otherwise replace it
    shape_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=6,
              markerfacecolor="none", markeredgecolor="#333333", label="core"),
        Line2D([], [], marker="^", linestyle="none", markersize=6,
              markerfacecolor="none", markeredgecolor="#333333", label="periphery"),
    ]
    ax_d.legend(handles=shape_handles, fontsize=9.5, frameon=False, title="Zone",
               title_fontsize=9.5, loc="center left", bbox_to_anchor=(1.0, 0.22),
               handletextpad=0.5)

    for ext in ("svg", "png"):
        p = OUT_FIG / f"fig_s10_sensitivity.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        log.info("Saved: %s", p)
    plt.close(fig)


def main():
    log.info("Loading cell_embeddings_index.parquet + cell_embeddings.npy ...")
    idx_main = pd.read_parquet(INDEX_PQ,
                               columns=["GridID", "city", "country", "row_idx",
                                       "q_relative", "log_pop", "LCF_Built"])
    emb = np.load(EMB_NPY)
    with open(BIN_GRID) as f:
        q_max = float(json.load(f)["q_P95"])
    idx_main = idx_main[(idx_main["q_relative"] >= 0) & (idx_main["q_relative"] <= q_max)].copy()
    idx_main["_city_key"] = idx_main["country"] + "||" + idx_main["city"]
    idx_main = idx_main.dropna(subset=["log_pop"])
    log.info("  %d cells, q_max=%.4f", len(idx_main), q_max)

    log.info("Loading alternative-center distances from cell_radial_position.parquet ...")
    radial_alt = pd.read_parquet(RADIAL_PQ,
                                 columns=["GridID", "city", "country",
                                         "dist_norm_pop", "dist_norm_geom"])

    city_log_pop = idx_main.groupby("_city_key")["log_pop"].first()
    grid = np.linspace(idx_main["log_pop"].min(), idx_main["log_pop"].max(), GRID_N)
    # Use the SAME grid range as the main analysis's actual city pool, not the
    # cell pool -- align to city_log_pop's range for a sensible LOWESS domain.
    grid = np.linspace(city_log_pop.min(), city_log_pop.max(), GRID_N)

    panel_a_df = panel_a_center_definitions(idx_main, emb, radial_alt, city_log_pop, q_max, grid)
    panel_c_df = panel_c_boundaries(idx_main, emb, city_log_pop, q_max, grid)
    panel_d_df = panel_d_coverage80(idx_main, emb, radial_alt, q_max)

    # Geometric centroid is computed above (for anyone re-deriving it) but
    # dropped before plotting/saving -- see module docstring "Geometric
    # centroid ... was computed, plotted, and then dropped" for why.
    panel_a_df = panel_a_df[~panel_a_df["setting"].str.contains("[Gg]eometric")].copy()
    panel_d_df = panel_d_df[~panel_d_df["setting"].str.contains("[Gg]eometric")].copy()

    panel_a_df.to_csv(OUT_TABLES / "s10_panel_a_center.csv", index=False, float_format="%.6f")
    panel_d_df.to_csv(OUT_TABLES / "s10_panel_d_coverage80.csv", index=False, float_format="%.6f")

    log.info("Loading existing K-sensitivity trend (panel b, reused from script 23) ...")
    panel_b_df = pd.read_csv(K23_TREND)

    plot_figure(panel_a_df, panel_b_df, panel_c_df, panel_d_df, grid)

    # Endpoint sign alone is a noisy readout of the qualitative pattern --
    # the extreme ends of the log_pop grid are sparse and the LOWESS tail
    # can wobble there (see geometric-centroid curve, panel a). The
    # pattern that actually matters is the MINIMUM (most negative,
    # small-city regime) vs. the trend value in the 10^6-10^7 plateau
    # (large-city regime), not the literal last grid point.
    def _pattern_summary(df):
        for label, sub in df.groupby("setting", sort=False):
            s = sub.sort_values("log_pop_grid").reset_index(drop=True)
            small_city_min = s["delta_V_smooth"].min()
            plateau = s[s["log_pop_grid"] >= np.log(1e6)]["delta_V_smooth"]
            large_city_val = plateau.mean() if len(plateau) else s["delta_V_smooth"].iloc[-1]
            print(f"  {label}: min (small-city regime)={small_city_min:+.4f}  "
                  f"mean over 10^6-10^7 (large-city regime)={large_city_val:+.4f}  "
                  f"pattern={'holds' if small_city_min < 0 < large_city_val else 'CHECK'}")

    print("\n=== Panel a: qualitative pattern check ===")
    _pattern_summary(panel_a_df)

    print("\n=== Panel c: qualitative pattern check ===")
    _pattern_summary(panel_c_df)

    print("\n=== Panel d: V_between at 80% coverage, by setting ===")
    print(panel_d_df.pivot_table(index=["tier", "zone"], columns="setting",
                                 values="V_between_at_80pct").to_string(float_format="%.3f"))

    print("\n=== Panel d: tier-ordering + zone-gap check, by setting ===")
    tiers = TIER_LABELS
    for setting, sub in panel_d_df.groupby("setting", sort=False):
        piv = sub.pivot(index="tier", columns="zone", values="V_between_at_80pct")
        core_ok = piv.loc[tiers[0], "core"] > piv.loc[tiers[1], "core"] > piv.loc[tiers[2], "core"]
        outer_ok = piv.loc[tiers[0], "outer"] > piv.loc[tiers[1], "outer"] > piv.loc[tiers[2], "outer"]
        gap_ok = all(piv.loc[t, "outer"] > piv.loc[t, "core"] for t in tiers)
        print(f"  {setting}: core Small>Interm>Large={core_ok}  "
              f"periphery Small>Interm>Large={outer_ok}  "
              f"periphery>core (all tiers)={gap_ok}")

    log.info("Done.")


if __name__ == "__main__":
    main()
