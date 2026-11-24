"""
20_trajectory_by_africapolis_tier.py
=======================================
Replaces the ad-hoc, data-fit population tercile/quartile splits used in
16/19 with the Africapolis (OECD/SWAC) size classes -- the standard,
externally-defensible framework for African city size, rather than a
boundary chosen by looking at our own curves.

Africapolis tiers (per-user adjustment: small tier's lower bound opened
to 0, since Africapolis's own 10k floor is a settlement-inclusion
criterion, not a meaningful lower cut for cities already in our n_cells
>= 50 sample):
  Small        Pop < 100,000
  Intermediate 100,000 <= Pop < 1,000,000
  Large        Pop >= 1,000,000

REVISED (methods review, same three issues already fixed in
21_trajectory_by_community.py -- ported here):

  (1) WEIGHTING MISMATCH (fixed): the composition profile previously
      pooled all cells directly (cell-weighted -- large cities would
      dominate a tier's reported composition), while the trajectory D(q)
      was already city-weighted. Both are now computed the same way:
      per-city type/group share per radial bin, THEN equal-city-weight
      averaged within the tier.

  (2) TAU MIS-NAMED: D(q) = D_max*(1-exp(-q/tau)) is monotonic with no
      inflection point. tau is the characteristic/e-folding radius --
      the q at which D(q) reaches 63.2% of D_max. Labels/outputs use
      "characteristic radius", not "inflection/transition point".

  (3) UNCERTAINTY TOO NARROW (fixed): tau_se was previously just
      sqrt(diag(pcov)) from curve_fit on 20 correlated radial-bin means.
      Replaced with a bootstrap (recompute D(q)/composition and refit
      D_max/tau each draw) for both the trajectory and the composition
      profile. A per-city bootstrap would ignore within-country
      correlation (cities in the same country share data-collection
      pipeline, administrative boundary conventions, and regional
      morphology) and understate the true continental uncertainty, so
      this uses a COUNTRY-block bootstrap instead: resample countries
      (within the tier) with replacement, each resampled country
      contributing its
      full set of qualifying cities (preserving resampling multiplicity).

  (4) FITTING-ORIGIN INCONSISTENCY (fixed): bin 0's D_q=0 is now an
      explicit (q=0, D=0) anchor added to the fit, not bin 0's
      slightly-off-zero midpoint treated as a real data point.

Output
------
  outputs/tables/trajectory_by_africapolis_tier.csv
      tier | radial_bin | q_mid | n_cities | D_q | D_q_lo | D_q_hi
  outputs/tables/group_profile_by_africapolis_tier.csv
      tier | radial_bin | q_mid | g_1..g_4 | g_1_lo..g_4_hi (city-weighted, bootstrap CI)
  outputs/tables/type_profile_by_africapolis_tier.csv
      tier | radial_bin | q_mid | p_1..p_17 | p_1_lo..p_17_hi (city-weighted, bootstrap CI)
  outputs/tables/trajectory_curve_fits_africapolis.csv
      tier | n_cities | D_max | D_max_lo | D_max_hi | tau | tau_lo | tau_hi | r_squared

Usage
-----
    python 20_trajectory_by_africapolis_tier.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).parent
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_ARRAYS = ROOT / "outputs" / "arrays"
INDEX_PQ   = OUT_TABLES / "cell_embeddings_index.parquet"
EMB_NPY    = OUT_ARRAYS / "cell_embeddings.npy"
BIN_GRID   = OUT_TABLES / "bin_grid.json"

N_BINS_FINE        = 20
N_TYPES             = 17
N_GROUPS            = 4
MIN_CELLS_PER_BIN   = 5
MIN_CITIES_PER_BIN  = 8
N_BOOT              = 500
BOOT_SEED           = 42

TIER_EDGES  = [0, 100_000, 1_000_000, np.inf]
TIER_LABELS = ["Small (<100k)", "Intermediate (100k-1M)", "Large (>=1M)"]
TIER_COLORS = {"Small (<100k)": "#8DA0CB",
              "Intermediate (100k-1M)": "#FFD700",
              "Large (>=1M)": "#C8292A"}

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
    """Fit anchored at the true origin (q=0, D=0); bin 0 excluded from
    the fitted pairs and replaced with an explicit (0,0) point."""
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
    """Per-city D_c(radial_bin) relative to the city's own core (bin 0)."""
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
    """Per-city composition share per radial bin -- city-weighted, same
    statistical object as the trajectory's D_c(q)."""
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
    out_rows = []
    for rb in range(N_BINS_FINE):
        q_mid = 0.5 * (q_edges[rb] + q_edges[rb + 1])
        sub = city_traj[city_traj["radial_bin"] == rb] if len(city_traj) else pd.DataFrame()
        n_c = sub["city_key"].nunique() if len(sub) else 0
        D_q = float(sub["D_c"].mean()) if n_c >= MIN_CITIES_PER_BIN else np.nan
        out_rows.append({"radial_bin": rb, "q_mid": q_mid, "n_cities": n_c, "D_q": D_q})
    return pd.DataFrame(out_rows)


def _aggregate_shares(city_shares: pd.DataFrame, q_edges: np.ndarray, n_cats: int) -> pd.DataFrame:
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
    city_keys = city_traj["city_key"].unique()
    if len(city_keys) < 10:
        return {"D_q_lo": np.full(N_BINS_FINE, np.nan),
               "D_q_hi": np.full(N_BINS_FINE, np.nan),
               "D_max_lo": np.nan, "D_max_hi": np.nan,
               "tau_lo": np.nan, "tau_hi": np.nan}

    D_q_draws = np.full((n_boot, N_BINS_FINE), np.nan)
    D_max_draws = np.full(n_boot, np.nan)
    tau_draws = np.full(n_boot, np.nan)

    # COUNTRY-block bootstrap (methods review round 2): resample countries
    # WITHIN this tier with replacement; each resampled country
    # contributes its FULL set of qualifying cities (preserving
    # resampling multiplicity), not an independent per-city resample --
    # see 23_scale_dependent_convergence.py's fix for why city-independent
    # resampling understates continental uncertainty.
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
    city_keys = city_shares["city_key"].unique()
    if len(city_keys) < 10:
        lo = {f"share_{k}_lo": np.full(N_BINS_FINE, np.nan) for k in range(1, n_cats + 1)}
        hi = {f"share_{k}_hi": np.full(N_BINS_FINE, np.nan) for k in range(1, n_cats + 1)}
        return {**lo, **hi}

    draws = {k: np.full((n_boot, N_BINS_FINE), np.nan) for k in range(1, n_cats + 1)}
    # COUNTRY-block bootstrap -- same design as _bootstrap_trajectory above.
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
                                   "log_pop", "label_k17"])
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
    idx = idx.dropna(subset=["log_pop"])

    city_pop = np.exp(idx.groupby("_city_key")["log_pop"].first())
    tier_map = pd.cut(city_pop, bins=TIER_EDGES, labels=TIER_LABELS, right=False)
    idx["_tier"] = idx["_city_key"].map(tier_map.to_dict())

    rng = np.random.default_rng(BOOT_SEED)
    traj_rows, fit_rows, type_rows, group_rows = [], [], [], []

    for tier in TIER_LABELS:
        sub = idx[idx["_tier"] == tier]
        n_cities_total = sub["_city_key"].nunique()
        pop_sub = city_pop[city_pop.index.isin(sub["_city_key"].unique())]
        log.info("=== %s: %d cities, pop range %.0f - %.0f ===",
                tier, n_cities_total, pop_sub.min() if len(pop_sub) else np.nan,
                pop_sub.max() if len(pop_sub) else np.nan)

        city_traj = _compute_city_trajectories(sub, emb)
        traj_df = _aggregate_trajectory(city_traj, q_edges)
        fit = _fit_curve_anchored(traj_df["q_mid"].values, traj_df["D_q"].values)
        log.info("  D(q) fit (origin-anchored): D_max=%.3f  tau=%.3f  R2=%.3f",
                fit["D_max"], fit["tau"], fit["r_squared"])
        log.info("  Bootstrapping (n=%d) ...", N_BOOT)
        boot = _bootstrap_trajectory(city_traj, q_edges, N_BOOT, rng)
        traj_df["D_q_lo"] = boot["D_q_lo"]
        traj_df["D_q_hi"] = boot["D_q_hi"]
        traj_df.insert(0, "tier", tier)
        traj_rows.append(traj_df)
        fit_rows.append({
            "tier": tier, "n_cities": n_cities_total,
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
        type_df.insert(0, "tier", tier)
        type_rows.append(type_df)

        city_group = _compute_city_shares(sub, "group4", N_GROUPS)
        group_df = _aggregate_shares(city_group, q_edges, N_GROUPS)
        boot_group = _bootstrap_shares(city_group, q_edges, N_GROUPS, N_BOOT, rng)
        for k in range(1, N_GROUPS + 1):
            group_df[f"g_{k}"] = group_df[f"share_{k}"]
            group_df[f"g_{k}_lo"] = boot_group[f"share_{k}_lo"]
            group_df[f"g_{k}_hi"] = boot_group[f"share_{k}_hi"]
        group_df = group_df.drop(columns=[f"share_{k}" for k in range(1, N_GROUPS + 1)])
        group_df.insert(0, "tier", tier)
        group_rows.append(group_df)

    traj_df_all = pd.concat(traj_rows, ignore_index=True)
    traj_df_all.to_csv(OUT_TABLES / "trajectory_by_africapolis_tier.csv",
                       index=False, float_format="%.6f")
    log.info("Saved: trajectory_by_africapolis_tier.csv (with bootstrap CI + origin-anchored fit)")

    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(OUT_TABLES / "trajectory_curve_fits_africapolis.csv",
                 index=False, float_format="%.6f")
    log.info("Saved: trajectory_curve_fits_africapolis.csv")
    print("\n=== D(q) curve fits by Africapolis tier (origin-anchored, bootstrap CI) ===")
    print(fit_df.to_string(index=False))

    type_df_all = pd.concat(type_rows, ignore_index=True)
    type_df_all.to_csv(OUT_TABLES / "type_profile_by_africapolis_tier.csv",
                       index=False, float_format="%.6f")
    log.info("Saved: type_profile_by_africapolis_tier.csv (city-weighted, with bootstrap CI)")

    group_df_all = pd.concat(group_rows, ignore_index=True)
    group_df_all.to_csv(OUT_TABLES / "group_profile_by_africapolis_tier.csv",
                        index=False, float_format="%.6f")
    log.info("Saved: group_profile_by_africapolis_tier.csv (city-weighted, with bootstrap CI)")

    log.info("Done.")


if __name__ == "__main__":
    main()
