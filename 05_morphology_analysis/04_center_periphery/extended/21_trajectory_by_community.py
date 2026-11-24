"""
21_trajectory_by_community.py
================================
Stratifies the core-periphery trajectory and composition profile by the
8 Leiden communities (02_regional_structure/community_detection) — the
regional/cultural grouping, not city size. Supersedes an earlier version
of this script after a methods review caught four issues, all fixed here:

  (1) WEIGHTING MISMATCH (fixed): the trajectory D(q) is computed per
      city first, then averaged across cities within a community
      (city-weighted). The composition profile previously pooled all
      cells directly (cell-weighted — large cities would dominate a
      community's reported composition). Both are now computed the same
      way: per-city type/group share per radial bin, THEN equal-city-
      weight averaged within the community. See _compute_city_shares().

  (2) TAU MIS-NAMED (fixed): D(q) = D_max*(1-exp(-q/tau)) is monotonic
      and has NO inflection point (no sign change in curvature). tau is
      the model's characteristic/e-folding radius — the q at which D(q)
      reaches 63.2% of its asymptote D_max, by definition. All labels/
      annotations now say "characteristic radius" and explicitly state
      the 63.2% definition; "inflection point" and "transition point" is
      no longer used.

  (3) UNCERTAINTY TOO NARROW (fixed): tau_se was previously just
      sqrt(diag(pcov)) from curve_fit on 20 correlated radial-bin means
      — this reflects curve-fit uncertainty GIVEN those 20 points, not
      between-CITY sampling uncertainty. Replaced with a bootstrap
      (recompute D(q) and refit D_max/tau each time) for both the
      trajectory and the composition profile, reported as 2.5/97.5
      percentile CIs. C7 (n=64 cities, R^2=0.77) and C4 (tau=0.50, not
      clearly within the observed q range) get the same bootstrap
      treatment so their wider uncertainty is visible rather than
      implied by a single understated se. A per-city bootstrap would
      ignore within-country correlation and understate continental
      uncertainty, so this uses a COUNTRY-block bootstrap instead:
      resample countries (within the community) with replacement, each
      resampled country contributing its full set of qualifying cities
      (preserving resampling multiplicity). This also makes the pooled
      (community==-1) curve's
      CI -- reused as the grey reference band in both
      25_trajectory_by_tier_combined.py and this script's own
      composition figure -- consistent with 20's tier-level CIs, which
      is load-bearing: before this fix the two used different
      uncertainty definitions side by side in the same figure.

      CAVEAT -- C7's CI band is visibly degenerate (narrow, jagged,
      almost flush with the point estimate on the lower side): C7 draws
      from only 3 countries (Egypt 58/64 cities = 91%, Mauritius 5,
      Algeria 1). Country-block resampling from a 3-country pool is
      close to resampling with only ~3 effective "balls in the urn", and
      since Egypt appears in the large majority of draws and dominates
      the pooled D(q) whenever it does, the resampled D(q) rarely
      deviates far below the point estimate -- the lower band is not
      wrong, it reflects a genuinely low-diversity resampling space, not
      a computation bug. C7's bootstrap CI should be read as unreliable/
      under-dispersed, not as evidence of unusually tight uncertainty;
      this is a sample-composition limitation of the community
      stratification itself (see scale_dependent_dispersion_within_
      community.csv or trajectory_curve_fits_by_community.csv for C7's
      n_cities=64), not something a different bootstrap implementation
      would fix.

  (4) FITTING-ORIGIN INCONSISTENCY (fixed): bin 0's D_q is set to 0 by
      construction (self-distance), but was previously fit at its bin
      MIDPOINT q (~0.02), not at q=0 — while the model D_max*(1-exp(-q/
      tau)) is exactly 0 at q=0 regardless of parameters. The curve was
      effectively being anchored by a slightly-off-origin point treated
      as true. Fixed: bin 0 is dropped from the fit's (q, D) pairs, and
      an explicit (q=0, D=0) anchor point is added instead — the
      correct origin, not a proxy for it.

Point (5) from the review (community differences may be confounded with
population-size composition differences, since different communities
have different city-size mixes and population independently predicts
the trajectory) is NOT addressed in this script — it needs a joint
model D_c(q) = f(q) + g(q)*log(Pop) + h_community(q) and is deferred to
a follow-up script, per the user's staged-fix plan (methods first here,
then the population-axis joint model, then figure redesign).

Output
------
  outputs/tables/trajectory_by_community.csv   (OVERWRITES 19's version
      with the origin-anchored fit + bootstrap CI columns added)
      community | radial_bin | q_mid | n_cities | D_q
  outputs/tables/trajectory_curve_fits_by_community.csv
      community | n_cities | D_max | D_max_lo | D_max_hi |
      tau | tau_lo | tau_hi | r_squared
  outputs/tables/type_profile_by_community.csv     (city-weighted now)
      community | radial_bin | q_mid | p_1..p_17 | p_1_lo..p_17_hi (boot CI)
  outputs/tables/group_profile_by_community.csv    (city-weighted now)
      community | radial_bin | q_mid | g_1..g_4 | g_1_lo..g_4_hi (boot CI)
  outputs/figures/trajectory_by_community_combined.svg/.png
  outputs/figures/composition_by_community.svg/.png

Usage
-----
    python 21_trajectory_by_community.py
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
from scipy.optimize import curve_fit

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

N_BINS_FINE        = 20
N_TYPES             = 17
N_GROUPS            = 4
MIN_CELLS_PER_BIN   = 5    # per-city, per-bin: need >=5 cells to count
MIN_CITIES_PER_BIN  = 8    # per-community, per-bin: need >=8 cities to report
N_BOOT              = 500
BOOT_SEED           = 42

PALETTE_9 = ["#f36424", "#672e92", "#8bcf02", "#139ed3",
            "#ff0000", "#ffd700", "#293890", "#74201e"]
POOLED_COLOR = "#555555"

PALETTE_17 = [
    "#DBDBDB", "#FEEFCF", "#FCBBC5", "#F5E6C8", "#E5152C",
    "#FFFF4C", "#5BA04A", "#FBB732", "#6F391F", "#B0E1E7",
    "#7B67EE", "#E89679", "#58C4EC", "#828F9B", "#1C1C5F",
    "#992FA3", "#C49A3C",
]
PALETTE_4 = {1: "#8DA0CB", 2: "#FC8D62", 3: "#66C2A5", 4: "#E78AC3"}

GROUP_MAP = {
    4: 1,  5: 1,
    1: 2,  3: 2,  6: 2,  7: 2,  8: 2,
    12: 3, 13: 3,
    2: 4,  9: 4, 10: 4, 11: 4, 14: 4, 15: 4, 16: 4, 17: 4,
}


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(1.0 - np.dot(a, b) / (na * nb))


def _exp_saturation(q, D_max, tau):
    return D_max * (1.0 - np.exp(-q / tau))


def _fit_curve_anchored(q_mid: np.ndarray, D_q: np.ndarray) -> dict:
    """
    Fit D(q) = D_max*(1-exp(-q/tau)), anchored at the true origin
    (q=0, D=0) rather than at bin 0's slightly-off-zero midpoint.
    bin-0 (D=0 by construction) is EXCLUDED from the fitted (q, D)
    pairs, and an explicit (0, 0) point is prepended instead — fixes
    issue (4) from the methods review.
    """
    mask = np.isfinite(q_mid) & np.isfinite(D_q) & (q_mid > 1e-6)
    if mask.sum() < 4:
        return {"D_max": np.nan, "tau": np.nan, "r_squared": np.nan}
    q = np.concatenate([[0.0], q_mid[mask]])
    D = np.concatenate([[0.0], D_q[mask]])
    try:
        popt, _ = curve_fit(_exp_saturation, q, D,
                            p0=[D.max() if D.max() > 0 else 0.3, 0.2],
                            bounds=([0, 1e-4], [5, 5]), maxfev=5000)
        D_max, tau = popt
        pred = _exp_saturation(q, *popt)
        ss_res = float(np.sum((D - pred) ** 2))
        ss_tot = float(np.sum((D - D.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"D_max": float(D_max), "tau": float(tau), "r_squared": r2}
    except Exception as e:
        log.warning("  Curve fit failed: %s", e)
        return {"D_max": np.nan, "tau": np.nan, "r_squared": np.nan}


def _compute_city_trajectories(idx_sub: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    """Per-city D_c(radial_bin) relative to the city's own core (bin 0).
    Returns long-form: city_key | radial_bin | D_c."""
    bin_means = {}
    for (ck, rb), rows in idx_sub.groupby(["_city_key", "radial_bin"])["row_idx"]:
        if len(rows) < MIN_CELLS_PER_BIN:
            continue
        bin_means[(ck, rb)] = emb[rows.values].mean(axis=0)

    rows_out = []
    for ck in idx_sub["_city_key"].unique():
        core_vec = bin_means.get((ck, 0))
        if core_vec is None:
            continue
        for rb in range(N_BINS_FINE):
            vec = bin_means.get((ck, rb))
            if vec is None:
                continue
            d = 0.0 if rb == 0 else _cos_dist(vec, core_vec)
            rows_out.append({"city_key": ck, "radial_bin": rb, "D_c": d})
    return pd.DataFrame(rows_out)


def _compute_city_shares(idx_sub: pd.DataFrame, group_col: str, n_cats: int) -> pd.DataFrame:
    """
    Per-city composition share (group_col in {label_k17 (1..17), group4
    (1..4)}) per radial bin — the per-city analogue of the trajectory's
    D_c(q), so composition and trajectory are the SAME statistical
    object (city-weighted), fixing issue (1) from the methods review.
    Returns long-form: city_key | radial_bin | share_1 .. share_n_cats.
    """
    rows_out = []
    for (ck, rb), sub in idx_sub.groupby(["_city_key", "radial_bin"]):
        if len(sub) < MIN_CELLS_PER_BIN:
            continue
        counts = np.bincount(sub[group_col].values.astype(int),
                             minlength=n_cats + 1)[1:n_cats + 1].astype(float)
        total = counts.sum()
        shares = counts / total if total > 0 else counts
        row = {"city_key": ck, "radial_bin": rb}
        row.update({f"share_{k+1}": shares[k] for k in range(n_cats)})
        rows_out.append(row)
    return pd.DataFrame(rows_out)


def _aggregate_trajectory(city_traj: pd.DataFrame, q_edges: np.ndarray) -> pd.DataFrame:
    """Equal-city-weight mean D_c(q) per bin -> population-level D(q)."""
    out_rows = []
    for rb in range(N_BINS_FINE):
        q_mid = 0.5 * (q_edges[rb] + q_edges[rb + 1])
        sub = city_traj[city_traj["radial_bin"] == rb] if len(city_traj) else pd.DataFrame()
        n_c = sub["city_key"].nunique() if len(sub) else 0
        D_q = float(sub["D_c"].mean()) if n_c >= MIN_CITIES_PER_BIN else np.nan
        out_rows.append({"radial_bin": rb, "q_mid": q_mid, "n_cities": n_c, "D_q": D_q})
    return pd.DataFrame(out_rows)


def _aggregate_shares(city_shares: pd.DataFrame, q_edges: np.ndarray, n_cats: int) -> pd.DataFrame:
    """Equal-city-weight mean share_k(q) per bin -> population-level profile."""
    out_rows = []
    for rb in range(N_BINS_FINE):
        q_mid = 0.5 * (q_edges[rb] + q_edges[rb + 1])
        sub = city_shares[city_shares["radial_bin"] == rb] if len(city_shares) else pd.DataFrame()
        n_c = sub["city_key"].nunique() if len(sub) else 0
        row = {"radial_bin": rb, "q_mid": q_mid, "n_cities": n_c}
        for k in range(1, n_cats + 1):
            col = f"share_{k}"
            row[col] = float(sub[col].mean()) if (n_c >= MIN_CITIES_PER_BIN and col in sub) else np.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _countries_in(city_keys: np.ndarray) -> dict:
    """city_key -> country, derived from the 'Country||City' key format."""
    return {ck: ck.split("||")[0] for ck in city_keys}


def _bootstrap_trajectory(city_traj: pd.DataFrame, q_edges: np.ndarray,
                          n_boot: int, rng: np.random.Generator) -> dict:
    """
    COUNTRY-block bootstrap (methods review round 2): resample countries
    WITHIN this stratum with replacement; each resampled country
    contributes its FULL set of qualifying cities (preserving resampling
    multiplicity), recompute the aggregated D(q) and refit D_max/tau each
    draw. Returns 2.5/97.5 percentile bands for D_q per bin, and for
    D_max/tau -- this is the between-COUNTRY-clustered uncertainty, not
    curve-fit se on 20 correlated bin means (fixes issue (3)) and not an
    independent-city resample that ignores within-country correlation
    (see 20_trajectory_by_africapolis_tier.py's and 23_scale_dependent_
    convergence.py's matching fix).
    """
    city_keys = city_traj["city_key"].unique()
    if len(city_keys) < 10:
        return {"D_q_lo": np.full(N_BINS_FINE, np.nan),
               "D_q_hi": np.full(N_BINS_FINE, np.nan),
               "D_max_lo": np.nan, "D_max_hi": np.nan,
               "tau_lo": np.nan, "tau_hi": np.nan}

    D_q_draws = np.full((n_boot, N_BINS_FINE), np.nan)
    D_max_draws = np.full(n_boot, np.nan)
    tau_draws = np.full(n_boot, np.nan)

    city_country = _countries_in(city_keys)
    countries = sorted(set(city_country.values()))
    cities_by_country = {}
    for ck in city_keys:
        cities_by_country.setdefault(city_country[ck], []).append(ck)
    grouped = {ck: g for ck, g in city_traj.groupby("city_key")}
    for b in range(n_boot):
        chosen_countries = rng.choice(countries, size=len(countries), replace=True)
        chosen_cities = [ck for c in chosen_countries for ck in cities_by_country[c]]
        if len(chosen_cities) < 10:
            continue
        resampled = pd.concat([grouped[ck] for ck in chosen_cities], ignore_index=True)
        agg = _aggregate_trajectory(resampled, q_edges)
        D_q_draws[b, :] = agg["D_q"].values
        fit = _fit_curve_anchored(agg["q_mid"].values, agg["D_q"].values)
        D_max_draws[b] = fit["D_max"]
        tau_draws[b] = fit["tau"]

    return {
        "D_q_lo": np.nanpercentile(D_q_draws, 2.5, axis=0),
        "D_q_hi": np.nanpercentile(D_q_draws, 97.5, axis=0),
        "D_max_lo": float(np.nanpercentile(D_max_draws, 2.5)),
        "D_max_hi": float(np.nanpercentile(D_max_draws, 97.5)),
        "tau_lo": float(np.nanpercentile(tau_draws, 2.5)),
        "tau_hi": float(np.nanpercentile(tau_draws, 97.5)),
    }


def _bootstrap_shares(city_shares: pd.DataFrame, q_edges: np.ndarray,
                      n_cats: int, n_boot: int, rng: np.random.Generator) -> dict:
    """COUNTRY-block bootstrap for the composition profile -- same
    resampling design as _bootstrap_trajectory above, so both CIs
    describe the same source of uncertainty."""
    city_keys = city_shares["city_key"].unique()
    if len(city_keys) < 10:
        lo = {f"share_{k}_lo": np.full(N_BINS_FINE, np.nan) for k in range(1, n_cats + 1)}
        hi = {f"share_{k}_hi": np.full(N_BINS_FINE, np.nan) for k in range(1, n_cats + 1)}
        return {**lo, **hi}

    draws = {k: np.full((n_boot, N_BINS_FINE), np.nan) for k in range(1, n_cats + 1)}
    city_country = _countries_in(city_keys)
    countries = sorted(set(city_country.values()))
    cities_by_country = {}
    for ck in city_keys:
        cities_by_country.setdefault(city_country[ck], []).append(ck)
    grouped = {ck: g for ck, g in city_shares.groupby("city_key")}
    for b in range(n_boot):
        chosen_countries = rng.choice(countries, size=len(countries), replace=True)
        chosen_cities = [ck for c in chosen_countries for ck in cities_by_country[c]]
        if len(chosen_cities) < 10:
            continue
        resampled = pd.concat([grouped[ck] for ck in chosen_cities], ignore_index=True)
        agg = _aggregate_shares(resampled, q_edges, n_cats)
        for k in range(1, n_cats + 1):
            draws[k][b, :] = agg[f"share_{k}"].values

    out = {}
    for k in range(1, n_cats + 1):
        out[f"share_{k}_lo"] = np.nanpercentile(draws[k], 2.5, axis=0)
        out[f"share_{k}_hi"] = np.nanpercentile(draws[k], 97.5, axis=0)
    return out


def main():
    log.info("Loading cell_embeddings_index.parquet + cell_embeddings.npy ...")
    idx = pd.read_parquet(INDEX_PQ,
                          columns=["city", "country", "row_idx", "q_relative",
                                   "community", "label_k17"])
    emb = np.load(EMB_NPY)
    with open(BIN_GRID) as f:
        q_max = float(json.load(f)["q_P95"])
    q_edges = np.linspace(0, q_max, N_BINS_FINE + 1)
    idx = idx[(idx["q_relative"] >= 0) & (idx["q_relative"] <= q_max)].copy()
    idx["radial_bin"] = np.clip(
        np.searchsorted(q_edges[1:-1], idx["q_relative"].values, side="right"),
        0, N_BINS_FINE - 1)
    idx["_city_key"] = idx["country"] + "||" + idx["city"]
    idx["group4"] = idx["label_k17"].astype(int).map(GROUP_MAP)
    communities = sorted(idx["community"].dropna().unique())

    rng = np.random.default_rng(BOOT_SEED)

    traj_rows, fit_rows = [], []
    type_rows, group_rows = [], []

    def _run_stratum(sub: pd.DataFrame, label: str, comm_val):
        n_cities_total = sub["_city_key"].nunique()
        log.info("=== %s: %d cities ===", label, n_cities_total)

        city_traj = _compute_city_trajectories(sub, emb)
        traj_df = _aggregate_trajectory(city_traj, q_edges)
        fit = _fit_curve_anchored(traj_df["q_mid"].values, traj_df["D_q"].values)
        log.info("  D(q) fit (origin-anchored): D_max=%.3f  tau=%.3f  R2=%.3f",
                fit["D_max"], fit["tau"], fit["r_squared"])
        log.info("  Bootstrapping (n=%d) ...", N_BOOT)
        boot = _bootstrap_trajectory(city_traj, q_edges, N_BOOT, rng)
        traj_df["D_q_lo"] = boot["D_q_lo"]
        traj_df["D_q_hi"] = boot["D_q_hi"]
        traj_df.insert(0, "community", comm_val)
        traj_rows.append(traj_df)
        fit_rows.append({
            "community": comm_val, "label": label, "n_cities": n_cities_total,
            "D_max": fit["D_max"], "D_max_lo": boot["D_max_lo"], "D_max_hi": boot["D_max_hi"],
            "tau": fit["tau"], "tau_lo": boot["tau_lo"], "tau_hi": boot["tau_hi"],
            "r_squared": fit["r_squared"],
        })

        city_type = _compute_city_shares(sub, "label_k17", N_TYPES)
        type_df = _aggregate_shares(city_type, q_edges, N_TYPES)
        boot_type = _bootstrap_shares(city_type, q_edges, N_TYPES, N_BOOT, rng)
        for k in range(1, N_TYPES + 1):
            type_df[f"p_{k}"] = type_df[f"share_{k}"]
            type_df[f"p_{k}_lo"] = boot_type[f"share_{k}_lo"]
            type_df[f"p_{k}_hi"] = boot_type[f"share_{k}_hi"]
        type_df = type_df.drop(columns=[f"share_{k}" for k in range(1, N_TYPES + 1)])
        type_df.insert(0, "community", comm_val)
        type_rows.append(type_df)

        city_group = _compute_city_shares(sub, "group4", N_GROUPS)
        group_df = _aggregate_shares(city_group, q_edges, N_GROUPS)
        boot_group = _bootstrap_shares(city_group, q_edges, N_GROUPS, N_BOOT, rng)
        for k in range(1, N_GROUPS + 1):
            group_df[f"g_{k}"] = group_df[f"share_{k}"]
            group_df[f"g_{k}_lo"] = boot_group[f"share_{k}_lo"]
            group_df[f"g_{k}_hi"] = boot_group[f"share_{k}_hi"]
        group_df = group_df.drop(columns=[f"share_{k}" for k in range(1, N_GROUPS + 1)])
        group_df.insert(0, "community", comm_val)
        group_rows.append(group_df)

    # Pooled (all cities) — same treatment, "community" = -1 sentinel
    _run_stratum(idx, "All cities (pooled)", -1)

    for comm in communities:
        comm = int(comm)
        _run_stratum(idx[idx["community"] == comm], f"C{comm+1}", comm)

    traj_all = pd.concat(traj_rows, ignore_index=True)
    traj_all.to_csv(OUT_TABLES / "trajectory_by_community.csv",
                    index=False, float_format="%.6f")
    log.info("Saved: trajectory_by_community.csv (with bootstrap CI + origin-anchored fit)")

    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(OUT_TABLES / "trajectory_curve_fits_by_community.csv",
                 index=False, float_format="%.6f")
    log.info("Saved: trajectory_curve_fits_by_community.csv")
    print("\n=== D(q) curve fits by community (origin-anchored, bootstrap CI) ===")
    print(fit_df.to_string(index=False))

    type_all = pd.concat(type_rows, ignore_index=True)
    type_all.to_csv(OUT_TABLES / "type_profile_by_community.csv",
                    index=False, float_format="%.6f")
    log.info("Saved: type_profile_by_community.csv (city-weighted, with bootstrap CI)")

    group_all = pd.concat(group_rows, ignore_index=True)
    group_all.to_csv(OUT_TABLES / "group_profile_by_community.csv",
                     index=False, float_format="%.6f")
    log.info("Saved: group_profile_by_community.csv (city-weighted, with bootstrap CI)")

    # ── Combined figure: the 8 fitted curves + pooled dashed reference +
    # end-of-curve labels, reporting ONE easy-to-read number per
    # community at the curve label -- D(q) evaluated at the
    # actual end of the observed range (q_max, the common-support cutoff
    # q_P95) rather than the extrapolated asymptote D_max, since D(q_max)
    # is a direct read of the fitted curve within the data's own support
    # and does not depend on how well the saturation model extrapolates
    # beyond it. Full D_max/tau/R2 + bootstrap CIs remain in
    # trajectory_curve_fits_by_community.csv for the companion table. ──
    log.info("Drawing single-panel trajectory summary figure ...")
    # Fixed absolute axes box (mm), NOT tight_layout -- so this figure's
    # axes width matches 23_scale_dependent_convergence.py's panel a
    # width exactly (both use AXES_W_MM/AXES_H_MM via the same
    # add_axes-in-mm pattern), independent of label text length. See
    # that script's matching block for the shared convention: font
    # sizes (9pt axis labels, 8pt ticks), axis colour (black), and
    # spine linewidth (0.8) are also unified across both figures.
    AXES_W_MM, AXES_H_MM = 100.0, 75.0
    LEFT_MM, RIGHT_MM, BOTTOM_MM, TOP_MM = 20.0, 60.0, 16.0, 6.0
    FIG_W_MM = LEFT_MM + AXES_W_MM + RIGHT_MM
    FIG_H_MM = BOTTOM_MM + AXES_H_MM + TOP_MM
    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
    ax = fig.add_axes([LEFT_MM / FIG_W_MM, BOTTOM_MM / FIG_H_MM,
                       AXES_W_MM / FIG_W_MM, AXES_H_MM / FIG_H_MM])
    # The fitted curve was previously drawn out to q_max (q_P95), but the
    # CI bands only cover the observed bins' q_mid values, whose last bin
    # centre sits half a bin-width short of q_max (bin IS an interval,
    # its centre is inside it, not at its right edge). That made the
    # curve visibly overshoot the end of its own CI band. Fixed: the
    # curve is now drawn only out to the last bin with data (same q
    # value the CI band already ends at), so curve and band terminate
    # together; the D(q) label value is still evaluated at q_max itself
    # (still a meaningful "end of common support" read), just not
    # plotted past where the data/CI actually stop.
    q_obs_end = float(traj_all[(traj_all["community"] == -1) &
                               traj_all["D_q"].notna()]["q_mid"].iloc[-1])
    q_grid = np.linspace(0, q_obs_end, 200)

    # Continental (pooled) line drawn noticeably thicker than the
    # per-community lines so it reads as the reference/background scale,
    # not just another one of the 8 -- community lines thinned slightly
    # to keep the overall figure from feeling heavier once CI bands are
    # added. CI bands are drawn on the actual observed bins (q_mid,
    # D_q_lo/D_q_hi from the country-block bootstrap), NOT interpolated
    # onto the smooth 200-point fit grid -- the bootstrap was only ever
    # run on the 20 real radial bins, so showing a band on the continuous
    # curve would imply a precision the resampling doesn't support there.
    pooled_fit = fit_df[fit_df["community"] == -1].iloc[0]
    pooled_traj = traj_all[traj_all["community"] == -1].dropna(subset=["D_q"])
    D_at_qmax_pooled = _exp_saturation(q_max, pooled_fit["D_max"], pooled_fit["tau"])
    ax.fill_between(pooled_traj["q_mid"], pooled_traj["D_q_lo"], pooled_traj["D_q_hi"],
                    color=POOLED_COLOR, alpha=0.18, linewidth=0, zorder=1)

    end_labels = []  # (comm, color, y_end, D_at_qmax) -- collect first,
                     # place after collision-avoidance (several curves
                     # converge near q_max)
    for comm in communities:
        comm = int(comm)
        color = PALETTE_9[comm % 9]
        fit_row = fit_df[fit_df["community"] == comm].iloc[0]
        if not np.isfinite(fit_row["tau"]):
            continue
        comm_traj = traj_all[traj_all["community"] == comm].dropna(subset=["D_q"])
        ax.fill_between(comm_traj["q_mid"], comm_traj["D_q_lo"], comm_traj["D_q_hi"],
                        color=color, alpha=0.07, linewidth=0, zorder=3)
        y_curve = _exp_saturation(q_grid, fit_row["D_max"], fit_row["tau"])
        D_at_qmax = _exp_saturation(q_max, fit_row["D_max"], fit_row["tau"])
        ax.plot(q_grid, y_curve, color=color, lw=1.3, zorder=4)
        end_labels.append((comm, color, y_curve[-1], D_at_qmax))

    # Pooled line drawn LAST, on top of every community curve/CI band
    # (highest zorder) -- it's the continental reference scale, not just
    # another one of the 8, and should never be occluded by them.
    ax.plot(q_grid, _exp_saturation(q_grid, pooled_fit["D_max"], pooled_fit["tau"]),
           color=POOLED_COLOR, lw=2.8, ls="--", zorder=10,
           label=f"Pooled, $D$({q_max:.1f})={D_at_qmax_pooled:.2f}")

    # De-collide end-of-curve labels: sort by y, then enforce a minimum
    # vertical gap by nudging each label up from the one below it.
    end_labels.sort(key=lambda t: t[2])
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0] or 1.0
    min_gap = 0.06 * y_range
    placed_y = []
    for comm, color, y_end, D_at_qmax in end_labels:
        y_label = y_end if not placed_y else max(y_end, placed_y[-1] + min_gap)
        placed_y.append(y_label)
        ax.text(q_grid[-1] * 1.02, y_label, f"C{comm+1}  $D$({q_max:.1f})={D_at_qmax:.2f}",
               color=color, fontsize=6.5, fontweight="bold",
               va="center", ha="left", clip_on=False)

    ax.set_xlabel("Relative radius $q = d / R_{98}$", fontsize=9)
    ax.set_ylabel("Embedding distance from own\npopulation-centred core, $D(q)$", fontsize=9)
    ax.set_xlim(0, q_max * 1.01)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("black")
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors="black", labelsize=8)

    # NOTE: no fig.tight_layout() here -- the axes box is already fixed
    # in absolute mm via add_axes() above (for cross-figure width
    # alignment with 23_scale_dependent_convergence.py); tight_layout
    # would override that fixed positioning.
    # Caption text wrapped onto explicit short lines -- fig.text does NOT
    # auto-wrap, so a single long string here forces bbox_inches="tight"
    # to widen the whole saved canvas to fit that one line, producing a
    # absurdly wide, short figure (confirmed: unwrapped version rendered
    # at 4277x1160px, aspect ratio 3.7:1, vs the ~140x100mm panel
    # actually drawn).
    caption = (
        r"$D(q_{max})$ is the fitted $D(q)=D_{max}(1-e^{-q/\tau})$ curve evaluated at "
        f"$q_{{max}}$={q_max:.2f}" "\n"
        "(the common-support cutoff, the actual end of the observed data range) -- "
        "a direct read within" "\n"
        r"the data's support, not the extrapolated asymptote $D_{max}$. Full $D_{max}$, "
        r"$\tau$ (characteristic" "\n"
        r"radius -- the $q$ at which $D(q)$ reaches 63.2% of $D_{max}$, not an inflection "
        "point) and their 95%" "\n"
        "country-block bootstrap CIs are reported in trajectory_curve_fits_by_community.csv. "
        "C7's CI band is visibly" "\n"
        "narrow/jagged because C7 draws from only 3 countries (Egypt=91% of its cities) -- "
        "an under-dispersed" "\n"
        "resampling space, not a tighter true uncertainty; interpret C7's band with caution."
    )
    fig.text(0.5, -0.02, caption, fontsize=5.5, color="#555555", style="italic",
            ha="center", va="top", linespacing=1.6)
    for ext in ("svg", "png"):
        out = OUT_FIG / f"trajectory_by_community_combined.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        log.info("Saved: %s", out)
    plt.close(fig)

    # ── Composition small multiples: 8 panels, one per community ──
    log.info("Drawing composition-by-community small multiples ...")
    fig2, axes2 = plt.subplots(4, 4, figsize=(220 / 25.4, 180 / 25.4))
    for comm in communities:
        comm = int(comm)
        row, col = divmod(comm, 2)
        ax17 = axes2[row, col * 2]
        ax4  = axes2[row, col * 2 + 1]

        t = type_all[type_all["community"] == comm]
        q = t["q_mid"].values
        shares17 = np.vstack([t[f"p_{k}"].values for k in range(1, N_TYPES + 1)])
        ax17.stackplot(q, shares17, colors=PALETTE_17, linewidth=0)
        ax17.set_xlim(q.min(), q.max())
        ax17.set_ylim(0, 1)
        ax17.set_title(f"C{comm+1} — 17 types", fontsize=7,
                       color=PALETTE_9[comm % 9], fontweight="bold")

        g = group_all[group_all["community"] == comm]
        qg = g["q_mid"].values
        shares4 = np.vstack([g[f"g_{k}"].values for k in range(1, N_GROUPS + 1)])
        colors4 = [PALETTE_4[k] for k in range(1, N_GROUPS + 1)]
        ax4.stackplot(qg, shares4, colors=colors4, linewidth=0)
        ax4.set_xlim(qg.min(), qg.max())
        ax4.set_ylim(0, 1)
        ax4.set_title(f"C{comm+1} — 4 groups", fontsize=7,
                      color=PALETTE_9[comm % 9], fontweight="bold")

        for ax in (ax17, ax4):
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=6)
            if row == 3:
                ax.set_xlabel("Relative radius $q$", fontsize=6.5)
        ax4.tick_params(labelleft=False)

    fig2.tight_layout(pad=1.0)
    for ext in ("svg", "png"):
        out = OUT_FIG / f"composition_by_community.{ext}"
        fig2.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        log.info("Saved: %s", out)
    plt.close(fig2)

    log.info("Done.")


if __name__ == "__main__":
    main()
