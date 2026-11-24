"""
23_scale_dependent_convergence.py
====================================
Core proposition: urban scale REVERSES the core-periphery geography of
BETWEEN-city morphological diversity. Small cities' cores differ more
from each other across cities than their peripheries do; large cities'
cores converge toward a common form across cities while their
peripheries diverge. This is a claim about CROSS-CITY similarity at a
given radial position, not about within-city (Shannon) diversity at
that position.

Method: a fixed-window-size sliding window (e.g.
300 cities/window) over only ~636 qualifying cities left too few
effectively-independent windows and, critically, does not generalize to
a per-community sensitivity check (some communities have as few as 64
qualifying cities total -- far fewer than one window). Replaced with a
K-NEAREST-NEIGHBOUR-BY-POPULATION design:

For each qualifying city c (population-based K-nearest-neighbours,
drawn from the full 636-city pool for the main analysis, or from
within c's own community for the sensitivity check):
    V_{c,core}  = mean_{c' in N_K(c)} [1 - cos(core_emb[c], core_emb[c'])]
    V_{c,outer} = mean_{c' in N_K(c)} [1 - cos(outer_emb[c], outer_emb[c'])]
    Delta_V_c   = V_{c,outer} - V_{c,core}
This gives EVERY qualifying city its own Delta_V_c and log(Pop_c) pair
-- no window-size trade-off, full use of all 636 cities, and the same
construction works identically whether neighbours are drawn from the
full pool or from one community. Delta_V_c vs log(Pop_c) is then
smoothed with a LOWESS trend (not a fixed window) for the headline
curve and its zero-crossing P*.

Core / outer zones (ABSOLUTE q): core: q < 0.2, outer: 0.75 < q <= q_P95.

SAMPLE COVERAGE (methods review, 2026-10-05): previously required
>=MIN_CELLS_ZONE=20 cells in BOTH zones to qualify -- diagnosed as a
severe, non-random sample restriction: of the 2504 cities with >=50
total cells (the project's base inclusion threshold, set in
00_define_size_tiers.py), this MIN_CELLS_ZONE=20 rule kept only 636
(25%), and the 1868 excluded cities were overwhelmingly small (median
pop 37k vs 159k for the included) and concentrated in specific countries
(Liberia, Mauritania, Burkina Faso, Somalia, DRC, South Sudan, Ethiopia
all had >85% of their cities excluded) -- NOT a random subset, a
systematic bias toward larger cities and certain countries. Fixed:
EVERY city with >=1 cell in BOTH zones now qualifies (only 11 of 2504
cities have zero outer-zone cells and are genuinely excluded); cities
with few cells in a zone contribute noisier per-city estimates, so
instead of an exclusion cutoff, each city gets a RELIABILITY WEIGHT
w_c = sqrt(min(n_core, n_cells in core, n_outer cells)) used when
aggregating across cities (LOWESS trend + country-block bootstrap) --
a low-n city still contributes a point, just a down-weighted one, not a
binary in/out decision. Weight validated empirically: a city's own
zone-mean embedding, estimated from just 1 cell, still has median cosine
similarity 0.845 to the "true" (40+-cell) mean for the same city
(see script development notes) -- small samples are noisy but not
uninformative, consistent with giving them reduced (not zero) weight.
KNN neighbour-finding and the per-pair cosine/JS distance calculation
are UNCHANGED -- weighting is applied only at the final LOWESS/bootstrap
aggregation step (statsmodels' lowess() has no native weight argument,
so frequency-weighting via integer row-repetition is used, a standard
treatment for non-negative integer weights).

K=10 is the main analysis; K=5 and K=15 are run as a sensitivity check --
if the trend and P* are stable across K, the result isn't an artifact of
the neighbourhood size.

Verification metric: same per-city construction, but on the 17-type
share vector via Jensen-Shannon distance instead of the embedding.

Output
------
  outputs/tables/scale_dependent_dispersion.csv
      city_key | log_pop | pop | V_E_core | V_E_outer | V_T_core | V_T_outer |
      delta_V_E | delta_V_T | K | reliability_weight
      (one row per qualifying city [2493 of 2504 -- only cities with zero
      cells in either zone excluded], for K=10; K=5/15 saved with a
      suffixed filename for the sensitivity check. reliability_weight =
      sqrt(min(n_core, n_outer)), used downstream for weighted LOWESS/
      bootstrap, not an inclusion filter.)
  outputs/tables/core_outer_reversal_trend.csv
      log_pop_grid | delta_V_E_smooth | delta_V_E_lo | delta_V_E_hi |
      delta_V_T_smooth | K
  outputs/tables/core_outer_reversal_P_star.csv
      K | P_star | P_star_lo | P_star_hi | n_cities
  outputs/figures/scale_dependent_core_periphery.svg/.png
      Two-panel: (a) per-city Delta_V_E scatter + LOWESS trend + bootstrap
      CI band, zero-crossing marked; (b) K=5/10/15 sensitivity overlay.

Usage
-----
    python 23_scale_dependent_convergence.py
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

N_TYPES        = 17
CORE_Q_MAX     = 0.2
OUTER_Q_MIN    = 0.75
MIN_CELLS_ZONE = 1     # inclusion floor (methods review: was 20, a severe
                       # non-random sample restriction -- see docstring);
                       # 1 excludes only genuinely-empty zones
K_MAIN         = 10
K_SENSITIVITY  = [5, 10, 15]
LOWESS_FRAC    = 0.5
N_BOOT         = 500
GRID_N         = 60   # points on the log_pop evaluation grid for the trend


def _pairwise_cos_dist_to_set(v: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Cosine distance from vector v to each row of V."""
    nv = np.linalg.norm(v)
    nV = np.linalg.norm(V, axis=1)
    nV[nV < 1e-12] = np.nan
    if nv < 1e-12:
        return np.full(len(V), np.nan)
    sims = (V @ v) / (nV * nv)
    return 1.0 - sims


def _js_dist_to_set(p: np.ndarray, P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Jensen-Shannon distance (base-2) from vector p to each row of P, vectorized."""
    p = p + eps
    p = p / p.sum()
    P = P + eps
    P = P / P.sum(axis=1, keepdims=True)
    M = 0.5 * (p[None, :] + P)
    logp, logP, logM = np.log2(p), np.log2(P), np.log2(M)
    kl_pm = np.sum(p[None, :] * (logp[None, :] - logM), axis=1)
    kl_Pm = np.sum(P * (logP - logM), axis=1)
    js_div = 0.5 * kl_pm + 0.5 * kl_Pm
    return np.sqrt(np.clip(js_div, 0, None))


def _knn_dispersion(city_keys, city_log_pop, core_emb, outer_emb,
                    core_type, outer_type, K) -> pd.DataFrame:
    """
    For each city in city_keys, find its K nearest neighbours BY
    log(Pop) within city_keys (excluding itself), then compute mean
    cosine/JS distance from that city to its neighbours, separately for
    core and outer. Neighbour pool = city_keys itself, so passing a
    community subset gives the within-community KNN sensitivity check
    "for free" via the same function.
    """
    keys = list(city_keys)
    n = len(keys)
    log_pops = np.array([city_log_pop[ck] for ck in keys])
    order = np.argsort(log_pops)
    sorted_keys = [keys[i] for i in order]
    sorted_pops = log_pops[order]

    core_mat = np.array([core_emb[ck] for ck in sorted_keys])
    outer_mat = np.array([outer_emb[ck] for ck in sorted_keys])
    core_type_mat = np.array([core_type[ck] for ck in sorted_keys])
    outer_type_mat = np.array([outer_type[ck] for ck in sorted_keys])

    rows = []
    for i, ck in enumerate(sorted_keys):
        # K nearest by log_pop rank distance (simple 1D neighbour search
        # on the sorted array -- exact nearest-K in |log_pop_i - log_pop_j|)
        dists = np.abs(sorted_pops - sorted_pops[i])
        dists[i] = np.inf  # exclude self
        knn_idx = np.argsort(dists)[:K]
        max_pop_gap = float(dists[knn_idx].max())

        v_e_core = float(np.nanmean(_pairwise_cos_dist_to_set(core_mat[i], core_mat[knn_idx])))
        v_e_outer = float(np.nanmean(_pairwise_cos_dist_to_set(outer_mat[i], outer_mat[knn_idx])))
        v_t_core = float(np.nanmean(_js_dist_to_set(core_type_mat[i], core_type_mat[knn_idx])))
        v_t_outer = float(np.nanmean(_js_dist_to_set(outer_type_mat[i], outer_type_mat[knn_idx])))

        rows.append({
            "city_key": ck, "log_pop": sorted_pops[i], "pop": float(np.exp(sorted_pops[i])),
            "V_E_core": v_e_core, "V_E_outer": v_e_outer,
            "V_T_core": v_t_core, "V_T_outer": v_t_outer,
            "max_log_pop_gap_to_knn": max_pop_gap, "K": K,
        })
    df = pd.DataFrame(rows)
    df["delta_V_E"] = df["V_E_outer"] - df["V_E_core"]
    df["delta_V_T"] = df["V_T_outer"] - df["V_T_core"]
    return df


def _lowess_trend(log_pop: np.ndarray, delta: np.ndarray, grid: np.ndarray,
                  frac: float = LOWESS_FRAC, weight: np.ndarray = None) -> np.ndarray:
    """
    LOWESS trend of delta vs log_pop. If weight is given (per-city
    reliability weight, e.g. sqrt(min(n_core, n_outer))), applies it as
    an integer FREQUENCY weight by row-repetition -- statsmodels' lowess()
    has no native weight argument, so a city with weight w contributes
    round(w) duplicate (log_pop, delta) rows, the standard treatment for
    non-negative integer/frequency weights in an estimator that doesn't
    accept them directly.
    """
    mask = np.isfinite(log_pop) & np.isfinite(delta)
    if weight is not None:
        mask &= np.isfinite(weight) & (weight > 0)
    if mask.sum() < 10:
        return np.full(len(grid), np.nan)
    x, y = log_pop[mask], delta[mask]
    if weight is not None:
        reps = np.maximum(np.round(weight[mask]).astype(int), 1)
        x = np.repeat(x, reps)
        y = np.repeat(y, reps)
    sm = lowess(y, x, frac=frac, return_sorted=True)
    return np.interp(grid, sm[:, 0], sm[:, 1])


def _zero_crossing(log_pop_grid: np.ndarray, delta_smooth: np.ndarray) -> float:
    sign_change = np.where(np.diff(np.sign(delta_smooth)) != 0)[0]
    if len(sign_change) == 0:
        return np.nan
    i = sign_change[0]
    x0, x1 = log_pop_grid[i], log_pop_grid[i + 1]
    y0, y1 = delta_smooth[i], delta_smooth[i + 1]
    # Direct linear interpolation -- np.interp(0, [y0, y1], [x0, x1]) silently
    # requires [y0, y1] to be increasing and gives a wrong answer (snaps to an
    # endpoint) when the curve crosses zero going from + to - rather than -
    # to + (methods review caught this; the K=10 main curve happens to cross
    # - to +, so the headline P* was unaffected, but a bootstrap draw's curve
    # can cross either way and was silently corrupting the P* CI).
    log_p_star = x0 - y0 * (x1 - x0) / (y1 - y0)
    return float(np.exp(log_p_star))


def main():
    log.info("Loading cell_embeddings_index.parquet + cell_embeddings.npy ...")
    idx = pd.read_parquet(INDEX_PQ,
                          columns=["city", "country", "row_idx", "q_relative",
                                   "log_pop", "label_k17", "community"])
    emb = np.load(EMB_NPY)
    with open(BIN_GRID) as f:
        q_max = float(json.load(f)["q_P95"])
    idx = idx[(idx["q_relative"] >= 0) & (idx["q_relative"] <= q_max)].copy()
    idx["_city_key"] = idx["country"] + "||" + idx["city"]
    idx = idx.dropna(subset=["log_pop"])
    log.info("  %d cells, q_max=%.4f", len(idx), q_max)

    # ── per-city core/outer mean embedding + type-share vector ──
    log.info("Computing per-city core/outer mean embedding and type-share vectors ...")
    core_sub = idx[idx["q_relative"] < CORE_Q_MAX]
    outer_sub = idx[(idx["q_relative"] > OUTER_Q_MIN) & (idx["q_relative"] <= q_max)]

    core_n = core_sub.groupby("_city_key").size()
    outer_n = outer_sub.groupby("_city_key").size()
    qualified = sorted(set(core_n[core_n >= MIN_CELLS_ZONE].index) &
                       set(outer_n[outer_n >= MIN_CELLS_ZONE].index))
    log.info("  %d cities qualify (>=%d cells in BOTH core and outer).",
            len(qualified), MIN_CELLS_ZONE)

    # Reliability weight per city: sqrt(min(n_core, n_outer)) -- used only
    # when aggregating across cities (LOWESS + bootstrap), not in the KNN
    # neighbour-finding or per-pair distance steps (see docstring).
    city_weight = {ck: float(np.sqrt(min(core_n[ck], outer_n[ck]))) for ck in qualified}
    log.info("  reliability weight: median=%.1f, min=%.1f, max=%.1f",
            np.median(list(city_weight.values())), min(city_weight.values()), max(city_weight.values()))

    city_log_pop = idx.groupby("_city_key")["log_pop"].first()
    city_community = idx.groupby("_city_key")["community"].first()

    core_emb, outer_emb, core_type, outer_type = {}, {}, {}, {}
    for ck in qualified:
        c_rows = core_sub[core_sub["_city_key"] == ck]
        o_rows = outer_sub[outer_sub["_city_key"] == ck]
        core_emb[ck] = emb[c_rows["row_idx"].values].mean(axis=0)
        outer_emb[ck] = emb[o_rows["row_idx"].values].mean(axis=0)

        c_counts = np.bincount(c_rows["label_k17"].values.astype(int),
                               minlength=N_TYPES + 1)[1:N_TYPES + 1].astype(float)
        core_type[ck] = c_counts / c_counts.sum() if c_counts.sum() > 0 else c_counts
        o_counts = np.bincount(o_rows["label_k17"].values.astype(int),
                               minlength=N_TYPES + 1)[1:N_TYPES + 1].astype(float)
        outer_type[ck] = o_counts / o_counts.sum() if o_counts.sum() > 0 else o_counts

    # ── main analysis: continental KNN dispersion, K=10 ──
    log.info("Computing continental KNN dispersion (K=%d) ...", K_MAIN)
    main_df = _knn_dispersion(qualified, city_log_pop, core_emb, outer_emb,
                              core_type, outer_type, K_MAIN)
    main_df["reliability_weight"] = main_df["city_key"].map(city_weight)
    main_df.to_csv(OUT_TABLES / "scale_dependent_dispersion.csv",
                   index=False, float_format="%.6f")
    log.info("Saved: scale_dependent_dispersion.csv (%d cities, K=%d)", len(main_df), K_MAIN)
    log.info("  max_log_pop_gap_to_knn: median=%.3f, p95=%.3f (smaller = tighter population match)",
            main_df["max_log_pop_gap_to_knn"].median(),
            main_df["max_log_pop_gap_to_knn"].quantile(0.95))

    # ── LOWESS trend + zero-crossing, K=10 ──
    # Weighted by reliability_weight (sqrt(min(n_core, n_outer))): a city
    # with few core/outer cells still contributes, just with less
    # influence on the trend, instead of the previous all-or-nothing
    # MIN_CELLS_ZONE=20 exclusion (see module docstring).
    log_pop_grid = np.linspace(main_df["log_pop"].min(), main_df["log_pop"].max(), GRID_N)
    delta_E_smooth = _lowess_trend(main_df["log_pop"].values, main_df["delta_V_E"].values, log_pop_grid,
                                   weight=main_df["reliability_weight"].values)
    delta_T_smooth = _lowess_trend(main_df["log_pop"].values, main_df["delta_V_T"].values, log_pop_grid,
                                   weight=main_df["reliability_weight"].values)
    P_star = _zero_crossing(log_pop_grid, delta_E_smooth)
    log.info("Delta_V_E (K=%d) LOWESS crosses zero near P* ~ %s",
            K_MAIN, f"{P_star:,.0f}" if np.isfinite(P_star) else "N/A")

    # ── country-block bootstrap for the K=10 trend + P* ──
    # FIXED (methods review, round 2): the previous version resampled
    # COUNTRIES and then re-ran _knn_dispersion() on the resampled city
    # list. But _knn_dispersion() finds each city's K-nearest-by-population
    # neighbours WITHIN the list it is given -- when a country is drawn
    # more than once, its cities appear multiple times in that list, and
    # those duplicate copies of the SAME city sit at population distance
    # EXACTLY ZERO from each other, so they become each other's nearest
    # "neighbours" with embedding distance exactly zero. This injects
    # spurious zero-distance pseudo-neighbours into the recomputed KNN
    # statistic for every city whose country got resampled >1 time (which
    # is most bootstrap draws, for most countries), corrupting delta_V_E
    # for those cities and anyone who has them as a real neighbour -- NOT
    # a legitimate consequence of preserving country resampling weight.
    #
    # Correct design: compute the per-city KNN dispersion statistic (V_E_
    # core, V_E_outer, delta_V_E) exactly ONCE, on the real (non-resampled)
    # 636-city sample -- this is `main_df`, already computed above. The
    # bootstrap then resamples COUNTRIES and, for each draw, takes the
    # already-computed per-city delta_V_E values for that draw's (possibly
    # repeated) city list and re-fits ONLY the LOWESS trend on them. No
    # KNN neighbourhoods are ever reconstructed on a resampled/duplicated
    # city list, so no city can become its own neighbour's duplicate.
    log.info("Bootstrapping (n=%d, country-block resample, LOWESS-only refit) ...", N_BOOT)
    rng = np.random.default_rng(42)
    main_df["country"] = main_df["city_key"].str.split(r"\|\|").str[0]
    countries = sorted(main_df["country"].unique())
    country_df = {c: main_df[main_df["country"] == c] for c in countries}

    boot_delta_E = np.full((N_BOOT, GRID_N), np.nan)
    boot_P_star = np.full(N_BOOT, np.nan)
    for b in range(N_BOOT):
        chosen_countries = rng.choice(countries, size=len(countries), replace=True)
        resampled = pd.concat([country_df[c] for c in chosen_countries], ignore_index=True)
        if len(resampled) < 20:
            continue
        d_smooth = _lowess_trend(resampled["log_pop"].values, resampled["delta_V_E"].values, log_pop_grid,
                                 weight=resampled["reliability_weight"].values)
        boot_delta_E[b] = d_smooth
        boot_P_star[b] = _zero_crossing(log_pop_grid, d_smooth)
        if (b + 1) % 100 == 0:
            log.info("  bootstrap draw %d/%d", b + 1, N_BOOT)

    delta_E_lo = np.nanpercentile(boot_delta_E, 2.5, axis=0)
    delta_E_hi = np.nanpercentile(boot_delta_E, 97.5, axis=0)
    P_star_lo = float(np.nanpercentile(boot_P_star, 2.5))
    P_star_hi = float(np.nanpercentile(boot_P_star, 97.5))
    log.info("P* (K=%d) bootstrap 95%% CI: [%s, %s]", K_MAIN,
            f"{P_star_lo:,.0f}" if np.isfinite(P_star_lo) else "N/A",
            f"{P_star_hi:,.0f}" if np.isfinite(P_star_hi) else "N/A")

    trend_rows = [{"K": K_MAIN, "log_pop_grid": log_pop_grid[i], "pop_grid": float(np.exp(log_pop_grid[i])),
                  "delta_V_E_smooth": delta_E_smooth[i], "delta_V_E_lo": delta_E_lo[i],
                  "delta_V_E_hi": delta_E_hi[i], "delta_V_T_smooth": delta_T_smooth[i]}
                 for i in range(GRID_N)]

    # ── K sensitivity: K=5 and K=15 (point estimate + LOWESS trend only, no bootstrap) ──
    log.info("Running K-sensitivity (K=%s) ...", K_SENSITIVITY)
    p_star_rows = [{"K": K_MAIN, "P_star": P_star, "P_star_lo": P_star_lo,
                   "P_star_hi": P_star_hi, "n_cities": len(qualified)}]
    sensitivity_curves = {K_MAIN: (log_pop_grid, delta_E_smooth)}
    for K in K_SENSITIVITY:
        if K == K_MAIN:
            continue
        df_k = _knn_dispersion(qualified, city_log_pop, core_emb, outer_emb,
                               core_type, outer_type, K)
        df_k["reliability_weight"] = df_k["city_key"].map(city_weight)
        smooth_k = _lowess_trend(df_k["log_pop"].values, df_k["delta_V_E"].values, log_pop_grid,
                                 weight=df_k["reliability_weight"].values)
        p_star_k = _zero_crossing(log_pop_grid, smooth_k)
        sensitivity_curves[K] = (log_pop_grid, smooth_k)
        p_star_rows.append({"K": K, "P_star": p_star_k, "P_star_lo": np.nan,
                           "P_star_hi": np.nan, "n_cities": len(qualified)})
        log.info("  K=%d: P* ~ %s", K, f"{p_star_k:,.0f}" if np.isfinite(p_star_k) else "N/A")
        for i in range(GRID_N):
            trend_rows.append({"K": K, "log_pop_grid": log_pop_grid[i],
                              "pop_grid": float(np.exp(log_pop_grid[i])),
                              "delta_V_E_smooth": smooth_k[i], "delta_V_E_lo": np.nan,
                              "delta_V_E_hi": np.nan, "delta_V_T_smooth": np.nan})

    trend_df = pd.DataFrame(trend_rows)
    trend_df.to_csv(OUT_TABLES / "core_outer_reversal_trend.csv", index=False, float_format="%.6f")
    log.info("Saved: core_outer_reversal_trend.csv")

    p_star_df = pd.DataFrame(p_star_rows)
    p_star_df.to_csv(OUT_TABLES / "core_outer_reversal_P_star.csv", index=False, float_format="%.6f")
    log.info("Saved: core_outer_reversal_P_star.csv")
    print("\n=== Zero-crossing population P* by K (neighbourhood size) ===")
    print(p_star_df.to_string(index=False))

    # ── within-community KNN sensitivity (community standardization as a
    # SENSITIVITY check, not a replacement for the
    # continental main result -- community is itself morphology-derived,
    # so fully "controlling" for it risks absorbing genuine regional
    # signal) ──
    log.info("Running within-community KNN sensitivity (K=%d) ...", K_MAIN)
    comm_rows = []
    for comm in sorted(city_community.dropna().unique()):
        comm = int(comm)
        comm_cities = [ck for ck in qualified if city_community.get(ck) == comm]
        if len(comm_cities) < K_MAIN + 5:
            log.warning("  C%d: only %d qualifying cities, skipping (need >= %d).",
                       comm + 1, len(comm_cities), K_MAIN + 5)
            continue
        df_c = _knn_dispersion(comm_cities, city_log_pop, core_emb, outer_emb,
                               core_type, outer_type, min(K_MAIN, len(comm_cities) - 1))
        df_c["community"] = comm
        df_c["reliability_weight"] = df_c["city_key"].map(city_weight)
        comm_rows.append(df_c)
        weighted_mean = np.average(df_c["delta_V_E"], weights=df_c["reliability_weight"])
        log.info("  C%d (n=%d): weighted mean delta_V_E=%.4f", comm + 1, len(comm_cities),
                weighted_mean)
    if comm_rows:
        comm_df = pd.concat(comm_rows, ignore_index=True)
        comm_df.to_csv(OUT_TABLES / "scale_dependent_dispersion_within_community.csv",
                       index=False, float_format="%.6f")
        log.info("Saved: scale_dependent_dispersion_within_community.csv")

    # ── figure ──
    # Fixed absolute axes boxes (mm), NOT tight_layout -- panel a's width
    # is set to exactly match 21_trajectory_by_community.py's single-panel
    # axes width (100mm) for direct visual stacking/comparison between the
    # two figures. Font sizes (9pt axis labels, 8pt ticks), axis colour
    # (black), and spine linewidth (0.8) are also unified with that script.
    log.info("Drawing figure ...")
    AXES_A_W_MM, AXES_H_MM = 100.0, 75.0
    AXES_B_W_MM = 65.0
    LEFT_A_MM, GAP_MM, RIGHT_B_MM = 20.0, 18.0, 8.0
    BOTTOM_MM, TOP_MM = 16.0, 6.0
    FIG_W_MM = LEFT_A_MM + AXES_A_W_MM + GAP_MM + AXES_B_W_MM + RIGHT_B_MM
    FIG_H_MM = BOTTOM_MM + AXES_H_MM + TOP_MM
    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
    ax_a = fig.add_axes([LEFT_A_MM / FIG_W_MM, BOTTOM_MM / FIG_H_MM,
                        AXES_A_W_MM / FIG_W_MM, AXES_H_MM / FIG_H_MM])
    ax_b_left = LEFT_A_MM + AXES_A_W_MM + GAP_MM
    ax_b = fig.add_axes([ax_b_left / FIG_W_MM, BOTTOM_MM / FIG_H_MM,
                        AXES_B_W_MM / FIG_W_MM, AXES_H_MM / FIG_H_MM])
    axes = [ax_a, ax_b]

    # Point size scales with each city's reliability weight (sqrt(min(n_core,
    # n_outer))) -- now that all 2493 qualifying cities are plotted (not just
    # the 636 that passed the old MIN_CELLS_ZONE=20 cutoff), this keeps
    # low-cell-count cities visually small/uncertain rather than equal-sized
    # dots indistinguishable from well-sampled cities.
    point_size = 3 + 10 * (main_df["reliability_weight"] / main_df["reliability_weight"].max())
    axes[0].scatter(main_df["pop"], main_df["delta_V_E"], s=point_size, alpha=0.18,
                    color="#293890", linewidth=0, zorder=2)
    axes[0].fill_between(np.exp(log_pop_grid), delta_E_lo, delta_E_hi,
                         color="#293890", alpha=0.15, linewidth=0, zorder=1)
    axes[0].plot(np.exp(log_pop_grid), delta_E_smooth, color="#293890", lw=2.0, zorder=3)
    axes[0].axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    # Label text deliberately does NOT report P_star as a precise point
    # value: a zero-crossing of a LOWESS trend is inherently less stable
    # than the trend shape itself, and the K=5/10/15 point estimates agree
    # tightly with each other (see printed p_star_df), which supports
    # stating a coarse reversal BAND (rounded CI bounds) rather than a
    # specific population. Computed dynamically from P_star_lo/hi (NOT
    # hardcoded) so this label never goes stale when the underlying
    # sample/weighting changes.
    if np.isfinite(P_star):
        axes[0].axvline(P_star, color="#C8292A", lw=1.0, ls=":")
        if np.isfinite(P_star_lo) and np.isfinite(P_star_hi):
            axes[0].axvspan(P_star_lo, P_star_hi, color="#C8292A", alpha=0.08, linewidth=0)
            def _round_sig(x, sig=1):
                if x <= 0 or not np.isfinite(x):
                    return x
                from math import floor, log10
                d = sig - int(floor(log10(abs(x)))) - 1
                return round(x, d)
            lo_label = _round_sig(P_star_lo)
            hi_label = _round_sig(P_star_hi)
            label_text = f" reversal around\n ~{lo_label:,.0f}-{hi_label:,.0f} pop."
        else:
            label_text = " reversal around\n ~%s pop." % f"{P_star:,.0f}"
        axes[0].text(P_star, axes[0].get_ylim()[1] * 0.85, label_text,
                    fontsize=6, color="#C8292A")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("City population", fontsize=9)
    axes[0].set_ylabel(r"$\Delta V_{E,c} = V_{E,outer} - V_{E,core}$" "\n(per-city, K=10 nearest by pop.)",
                       fontsize=9)
    axes[0].set_title("a", loc="left", fontsize=9, fontweight="bold")

    k_colors = {5: "#8DA0CB", 10: "#293890", 15: "#C8292A"}
    for K in K_SENSITIVITY:
        grid_k, smooth_k = sensitivity_curves[K]
        axes[1].plot(np.exp(grid_k), smooth_k, color=k_colors.get(K, "#555555"),
                    lw=1.8, label=f"K={K}")
    axes[1].axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("City population", fontsize=9)
    axes[1].set_ylabel(r"$\Delta V_E(P)$ LOWESS trend", fontsize=9)
    axes[1].legend(fontsize=6.5, frameon=False, title="Neighbourhood size", title_fontsize=6.5)
    axes[1].set_title("b", loc="left", fontsize=9, fontweight="bold")

    for ax in axes:
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color("black")
            ax.spines[sp].set_linewidth(0.8)
        ax.tick_params(colors="black", labelsize=8)

    # NOTE: no fig.tight_layout() here -- axes boxes are fixed in
    # absolute mm via add_axes() above (for cross-figure width alignment
    # with 21_trajectory_by_community.py's panel).
    for ext in ("svg", "png"):
        out = OUT_FIG / f"scale_dependent_core_periphery.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        log.info("Saved: %s", out)
    plt.close(fig)

    log.info("Done.")


if __name__ == "__main__":
    main()
