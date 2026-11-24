"""
30_tier_radial_composition_divergence.py
===========================================
Fills a structural gap: the community branch has both a trajectory
(21_trajectory_by_community.py) AND a composition explanation
(composition_by_community.svg) for why communities differ, but the
city-SIZE branch only has a trajectory (trajectory_by_tier_combined.svg)
-- no answer for WHICH morphology types drive the size-tier differences,
or WHERE along the radial profile (core vs periphery) those differences
are concentrated.

Research question
------------------
At a given radial position q, how different is Small cities' average
17-type composition from Large cities' average 17-type composition --
and does that cross-tier compositional gap concentrate at the core or
the periphery?

    J_ab(q) = JS[ p_a(q), p_b(q) ]

where p_a(q) is tier a's city-weighted mean 17-type share vector at
radial bin q (already computed, city-weighted, by
20_trajectory_by_africapolis_tier.py's _compute_city_shares /
_aggregate_shares -- reused directly here via import, not recomputed
from scratch). Three pairwise comparisons: Small-Intermediate,
Intermediate-Large, Small-Large.

This is NOT the same statistical object as 29_core_outer_divergence_
mechanism.py: 29 compares INDIVIDUAL same-sized cities to each other
(within-tier dispersion); this script compares tiers' AVERAGE profiles
to each other (between-tier central tendency), at every radial position,
not just core/outer. Nor is it the same as 02_regional_structure's
05_community_profiles.py, which profiles Leiden communities (a
region/morphology grouping, not a size grouping) and has no radial
dimension at all -- every city there contributes ONE whole-city
composition vector, not a per-q curve.

Preliminary (non-bootstrapped) point estimates, verified against the
existing tier composition tables:
    Comparison              core q~0.02   outer q~0.91   decline
    Small-Intermediate         0.433         0.262          39%
    Intermediate-Large         0.201         0.125          38%
    Small-Large                 0.560         0.310          45%
i.e. cross-tier compositional divergence is concentrated at the CORE and
narrows substantially toward the periphery -- city size mainly reshapes
which morphology dominates the core, while peripheries across size tiers
look more alike. The 4-group version shows the same direction (0.338 ->
0.264) but a weaker decline, meaning the effect lives disproportionately
in the 17-type (not the 4-group) resolution.

Bootstrap design (critical correctness point)
-----------------------------------------------
Each bootstrap draw resamples COUNTRIES ONCE and applies that SAME
resampled country set to recompute ALL THREE tiers' composition profiles
before computing the three JS distances for that draw. Resampling each
tier independently would let different draws' Small/Intermediate/Large
profiles be built from different, uncorrelated country subsets --
artificially injecting country-composition noise into the JS distance
that has nothing to do with genuine size-tier differentiation. (This is
a different bootstrap correctness issue from 23/27/20's "duplicate
cities in the resample becoming each other's neighbours" bug -- here the
risk is INDEPENDENT resampling across groups that must share one
resampled country set, not recomputation on duplicated rows.)

17-type is the primary analysis; the 4-group version is a robustness
check (same script, both computed).

Output
------
  outputs/tables/tier_radial_composition_divergence.csv
      q_mid | radial_bin | pair | resolution | n_types | JS | JS_lo | JS_hi
  outputs/tables/tier_core_outer_type_dumbbell.csv
      label_k17 | type_name | share_small_core | share_large_core |
      abs_diff_core | share_small_outer | share_large_outer | abs_diff_outer
  outputs/figures/tier_radial_composition_divergence.svg/.png

Usage
-----
    python 30_tier_radial_composition_divergence.py
"""

import importlib.util
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
BIN_GRID   = OUT_TABLES / "bin_grid.json"
OUT_FIG    = ROOT / "outputs" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

# Import 20_trajectory_by_africapolis_tier.py by path (filename starts
# with a digit, can't be a normal import statement) -- reuses its
# _compute_city_shares / _aggregate_shares / constants directly rather
# than duplicating that aggregation logic.
_spec = importlib.util.spec_from_file_location(
    "tier20", ROOT / "20_trajectory_by_africapolis_tier.py")
tier20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tier20)

N_BINS_FINE = tier20.N_BINS_FINE
N_TYPES     = tier20.N_TYPES
N_GROUPS    = tier20.N_GROUPS
TIER_EDGES  = tier20.TIER_EDGES
TIER_LABELS = tier20.TIER_LABELS
GROUP_MAP   = tier20.GROUP_MAP
MIN_CITIES_PER_BIN = tier20.MIN_CITIES_PER_BIN

N_BOOT    = 500
BOOT_SEED = 42
TOP_N_DUMBBELL = 8

TIER_PAIRS = [
    ("Small (<100k)", "Intermediate (100k-1M)", "Small-Intermediate"),
    ("Intermediate (100k-1M)", "Large (>=1M)", "Intermediate-Large"),
    ("Small (<100k)", "Large (>=1M)", "Small-Large"),
]
PAIR_COLORS = {"Small-Intermediate": "#8DA0CB", "Intermediate-Large": "#FFD700",
              "Small-Large": "#C8292A"}


def _js_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    p = p / p.sum()
    q = np.asarray(q, dtype=float) + eps
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(np.sqrt(max(0.5 * kl_pm + 0.5 * kl_qm, 0.0)))


def _tier_shares_from_profile(profile_df: pd.DataFrame, n_cats: int) -> dict:
    """radial_bin -> np.array of n_cats shares (NaN array if below MIN_CITIES_PER_BIN)."""
    out = {}
    for rb in range(N_BINS_FINE):
        row = profile_df[profile_df["radial_bin"] == rb]
        if len(row) == 0 or row["n_cities"].iloc[0] < MIN_CITIES_PER_BIN:
            out[rb] = None
            continue
        cols = [f"share_{k}" for k in range(1, n_cats + 1)]
        out[rb] = row[cols].values[0].astype(float)
    return out


def _compute_js_curve(idx: pd.DataFrame, q_edges: np.ndarray, group_col: str,
                      n_cats: int) -> dict:
    """For the real (non-resampled) data: per-tier city-shares table,
    per-tier aggregated profile, and per-bin JS distance for each pair."""
    tier_city_shares = {}
    tier_profile = {}
    for tier in TIER_LABELS:
        sub = idx[idx["_tier"] == tier]
        city_shares = tier20._compute_city_shares(sub, group_col, n_cats)
        tier_city_shares[tier] = city_shares
        tier_profile[tier] = tier20._aggregate_shares(city_shares, q_edges, n_cats)

    tier_bin_shares = {t: _tier_shares_from_profile(tier_profile[t], n_cats)
                       for t in TIER_LABELS}

    js_rows = []
    for rb in range(N_BINS_FINE):
        q_mid = 0.5 * (q_edges[rb] + q_edges[rb + 1])
        for a, b, pair_name in TIER_PAIRS:
            pa, pb = tier_bin_shares[a][rb], tier_bin_shares[b][rb]
            js = _js_distance(pa, pb) if (pa is not None and pb is not None) else np.nan
            js_rows.append({"radial_bin": rb, "q_mid": q_mid, "pair": pair_name, "JS": js})
    return {"tier_city_shares": tier_city_shares, "js_df": pd.DataFrame(js_rows)}


def _bootstrap_js(idx: pd.DataFrame, q_edges: np.ndarray, group_col: str,
                  n_cats: int, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Country-block bootstrap shared ACROSS all three tiers: each draw
    resamples the full country pool ONCE, then rebuilds all three tiers'
    composition profiles restricted to that SAME resampled country set,
    before computing the three pairwise JS curves for that draw. This is
    the correctness-critical design: resampling tiers independently would
    let different draws mix uncorrelated country subsets across tiers,
    injecting spurious country-composition
    noise into the JS distance.
    """
    idx = idx.dropna(subset=["_tier"])
    countries = sorted(idx["country"].unique())
    country_rows = {c: idx[idx["country"] == c] for c in countries}

    boot_js = {pair_name: np.full((n_boot, N_BINS_FINE), np.nan)
              for _, _, pair_name in TIER_PAIRS}

    for b in range(n_boot):
        chosen_countries = rng.choice(countries, size=len(countries), replace=True)
        resampled = pd.concat([country_rows[c] for c in chosen_countries], ignore_index=True)
        if len(resampled) < 50:
            continue

        tier_bin_shares = {}
        for tier in TIER_LABELS:
            sub = resampled[resampled["_tier"] == tier]
            if sub["_city_key"].nunique() < MIN_CITIES_PER_BIN:
                tier_bin_shares[tier] = {rb: None for rb in range(N_BINS_FINE)}
                continue
            city_shares = tier20._compute_city_shares(sub, group_col, n_cats)
            profile = tier20._aggregate_shares(city_shares, q_edges, n_cats)
            tier_bin_shares[tier] = _tier_shares_from_profile(profile, n_cats)

        for rb in range(N_BINS_FINE):
            for a, b_tier, pair_name in TIER_PAIRS:
                pa, pb = tier_bin_shares[a][rb], tier_bin_shares[b_tier][rb]
                if pa is not None and pb is not None:
                    boot_js[pair_name][b, rb] = _js_distance(pa, pb)

        if (b + 1) % 100 == 0:
            log.info("  bootstrap draw %d/%d", b + 1, n_boot)

    out_rows = []
    for _, _, pair_name in TIER_PAIRS:
        lo = np.nanpercentile(boot_js[pair_name], 2.5, axis=0)
        hi = np.nanpercentile(boot_js[pair_name], 97.5, axis=0)
        for rb in range(N_BINS_FINE):
            out_rows.append({"radial_bin": rb, "pair": pair_name,
                            "JS_lo": lo[rb], "JS_hi": hi[rb]})
    return pd.DataFrame(out_rows)


def main():
    log.info("Loading cell_embeddings_index.parquet ...")
    idx = pd.read_parquet(INDEX_PQ,
                          columns=["city", "country", "row_idx", "q_relative",
                                   "log_pop", "label_k17"])
    with open(BIN_GRID) as f:
        q_max = float(json.load(f)["q_P95"])
    q_edges = np.linspace(0, q_max, N_BINS_FINE + 1)
    idx = idx[(idx["q_relative"] >= 0) & (idx["q_relative"] <= q_max)].copy()
    idx["radial_bin"] = np.clip(
        np.searchsorted(q_edges[1:-1], idx["q_relative"].values, side="right"),
        0, N_BINS_FINE - 1)
    idx["_city_key"] = idx["country"] + "||" + idx["city"]
    idx["group4"] = idx["label_k17"].astype(int).map(GROUP_MAP)
    idx = idx.dropna(subset=["log_pop"])

    city_pop = np.exp(idx.groupby("_city_key")["log_pop"].first())
    tier_map = pd.cut(city_pop, bins=TIER_EDGES, labels=TIER_LABELS, right=False)
    idx["_tier"] = idx["_city_key"].map(tier_map.to_dict())

    all_rows = []
    dumbbell_df = None
    for group_col, n_cats, resolution in [("label_k17", N_TYPES, "17-type"),
                                          ("group4", N_GROUPS, "4-group")]:
        log.info("=== Resolution: %s ===", resolution)
        result = _compute_js_curve(idx, q_edges, group_col, n_cats)
        js_df = result["js_df"]
        js_df["resolution"] = resolution
        js_df["n_types"] = n_cats

        log.info("  Point estimates (no CI yet):")
        for _, _, pair_name in TIER_PAIRS:
            sub = js_df[js_df["pair"] == pair_name].sort_values("q_mid")
            core_js = sub["JS"].iloc[0]
            outer_js = sub["JS"].iloc[-1]
            decline = (core_js - outer_js) / core_js * 100 if core_js > 0 else np.nan
            log.info("    %s: core=%.3f outer=%.3f decline=%.0f%%",
                    pair_name, core_js, outer_js, decline)

        log.info("  Bootstrapping (n=%d, country set shared across tiers) ...", N_BOOT)
        rng = np.random.default_rng(BOOT_SEED)
        boot_df = _bootstrap_js(idx, q_edges, group_col, n_cats, N_BOOT, rng)
        js_df = js_df.merge(boot_df, on=["radial_bin", "pair"], how="left")
        all_rows.append(js_df)

        if resolution == "17-type":
            # ── dumbbell data: Small vs Large, core and outer, 17 types ──
            tier_city_shares = result["tier_city_shares"]
            small_profile = tier20._aggregate_shares(tier_city_shares["Small (<100k)"], q_edges, N_TYPES)
            large_profile = tier20._aggregate_shares(tier_city_shares["Large (>=1M)"], q_edges, N_TYPES)
            core_rb, outer_rb = 0, N_BINS_FINE - 1
            small_core = small_profile[small_profile["radial_bin"] == core_rb].iloc[0]
            large_core = large_profile[large_profile["radial_bin"] == core_rb].iloc[0]
            small_outer = small_profile[small_profile["radial_bin"] == outer_rb].iloc[0]
            large_outer = large_profile[large_profile["radial_bin"] == outer_rb].iloc[0]
            dumbbell_rows = []
            for k in range(1, N_TYPES + 1):
                s_core = float(small_core[f"share_{k}"])
                l_core = float(large_core[f"share_{k}"])
                s_outer = float(small_outer[f"share_{k}"])
                l_outer = float(large_outer[f"share_{k}"])
                dumbbell_rows.append({
                    "label_k17": k,
                    "share_small_core": s_core, "share_large_core": l_core,
                    "abs_diff_core": abs(l_core - s_core),
                    "share_small_outer": s_outer, "share_large_outer": l_outer,
                    "abs_diff_outer": abs(l_outer - s_outer),
                })
            dumbbell_df = pd.DataFrame(dumbbell_rows)

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df.to_csv(OUT_TABLES / "tier_radial_composition_divergence.csv",
                   index=False, float_format="%.6f")
    log.info("Saved: tier_radial_composition_divergence.csv")

    dumbbell_df.to_csv(OUT_TABLES / "tier_core_outer_type_dumbbell.csv",
                       index=False, float_format="%.6f")
    log.info("Saved: tier_core_outer_type_dumbbell.csv")

    # ── figure: two panels, no heatmap ──
    log.info("Drawing figure ...")
    AXES_A_W_MM, AXES_H_MM = 85.0, 75.0
    AXES_B_W_MM = 75.0
    LEFT_A_MM, GAP_MM, RIGHT_B_MM = 20.0, 20.0, 10.0
    BOTTOM_MM, TOP_MM = 16.0, 6.0
    FIG_W_MM = LEFT_A_MM + AXES_A_W_MM + GAP_MM + AXES_B_W_MM + RIGHT_B_MM
    FIG_H_MM = BOTTOM_MM + AXES_H_MM + TOP_MM
    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
    ax_a = fig.add_axes([LEFT_A_MM / FIG_W_MM, BOTTOM_MM / FIG_H_MM,
                        AXES_A_W_MM / FIG_W_MM, AXES_H_MM / FIG_H_MM])
    ax_b_left = LEFT_A_MM + AXES_A_W_MM + GAP_MM
    ax_b = fig.add_axes([ax_b_left / FIG_W_MM, BOTTOM_MM / FIG_H_MM,
                        AXES_B_W_MM / FIG_W_MM, AXES_H_MM / FIG_H_MM])

    # Panel a: radial composition divergence, 17-type (primary)
    t17 = full_df[full_df["resolution"] == "17-type"]
    for a, b_tier, pair_name in TIER_PAIRS:
        sub = t17[t17["pair"] == pair_name].sort_values("q_mid")
        color = PAIR_COLORS[pair_name]
        ax_a.fill_between(sub["q_mid"], sub["JS_lo"], sub["JS_hi"],
                          color=color, alpha=0.15, linewidth=0, zorder=1)
        ax_a.plot(sub["q_mid"], sub["JS"], color=color, lw=1.8, zorder=3, label=pair_name)

    sl = t17[t17["pair"] == "Small-Large"].sort_values("q_mid")
    core_js, outer_js = sl["JS"].iloc[0], sl["JS"].iloc[-1]
    decline_pct = (core_js - outer_js) / core_js * 100
    ax_a.text(0.97, 0.72,
             f"Small-Large: {core_js:.2f} $\\rightarrow$ {outer_js:.2f} (-{decline_pct:.0f}%)",
             transform=ax_a.transAxes, fontsize=6.5, color=PAIR_COLORS["Small-Large"],
             ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", linewidth=0.5))

    ax_a.set_xlabel("Relative radius $q = d/R_{98}$", fontsize=9)
    ax_a.set_ylabel("Cross-tier composition divergence\n(Jensen-Shannon distance, 17 types)", fontsize=9)
    ax_a.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax_a.set_title("a", loc="left", fontsize=9, fontweight="bold")

    # Panel b: dumbbell, top types by core |diff|, Small vs Large
    top_types = dumbbell_df.sort_values("abs_diff_core", ascending=False).head(TOP_N_DUMBBELL)
    top_types = top_types.sort_values("abs_diff_core")
    y_pos = np.arange(len(top_types))
    for i, (_, row) in enumerate(top_types.iterrows()):
        ax_b.plot([row["share_small_core"], row["share_large_core"]], [y_pos[i], y_pos[i]],
                 color="#AAAAAA", lw=1.2, zorder=1)
        ax_b.scatter(row["share_small_core"], y_pos[i], color=PAIR_COLORS["Small-Intermediate"],
                    s=35, zorder=3, label="Small (core)" if i == 0 else None)
        ax_b.scatter(row["share_large_core"], y_pos[i], color=PAIR_COLORS["Small-Large"],
                    s=35, zorder=3, label="Large (core)" if i == 0 else None)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([f"T{int(t)}" for t in top_types["label_k17"]], fontsize=7)
    ax_b.set_xlabel("Share of core cells", fontsize=9)
    ax_b.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax_b.set_title("b", loc="left", fontsize=9, fontweight="bold")
    ax_b.set_ylim(-0.7, len(top_types) - 0.3)

    for ax in (ax_a, ax_b):
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color("black")
            ax.spines[sp].set_linewidth(0.8)
        ax.tick_params(colors="black", labelsize=8)

    caption = (
        "Panel a: Jensen-Shannon distance between size-tier pairs' city-weighted 17-type "
        "composition at each" "\n"
        "radial position (reuses 20_trajectory_by_africapolis_tier.py's composition tables "
        "-- no new metric)." "\n"
        "Bands: 95% bootstrap CI, countries resampled ONCE per draw and shared across all "
        "three tiers" "\n"
        "(tiers resampled independently would inject spurious country-composition noise). "
        "Panel b: the" "\n"
        f"{TOP_N_DUMBBELL} types with the largest Small-vs-Large core share gap (T1-T17 = "
        "label_k17; full type names in" "\n"
        "tier_core_outer_type_dumbbell.csv). Outer-position dumbbell and 4-group robustness "
        "check in the" "\n"
        "corresponding CSV, not shown here."
    )
    fig.text(0.5, -0.02, caption, fontsize=5.5, color="#555555", style="italic",
            ha="center", va="top", linespacing=1.6)

    for ext in ("svg", "png"):
        out = OUT_FIG / f"tier_radial_composition_divergence.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        log.info("Saved: %s", out)
    plt.close(fig)

    log.info("Done.")


if __name__ == "__main__":
    main()
