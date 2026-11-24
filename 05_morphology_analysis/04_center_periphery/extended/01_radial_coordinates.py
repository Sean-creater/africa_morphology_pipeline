"""
01_radial_coordinates.py
========================
Q4: Internal spatial organisation of urban morphology across city-size regimes.

Compute radial bin assignments for every cell, using three distance
coordinates as agreed:

  Main          scaled distance   u  = dist_m_primary / Pop^alpha
  Robustness    relative radius   q  = dist_m_primary / R98_m
  Supplement    absolute km       d  = dist_km_primary

Radial bins
-----------
  10 equal-width bins within the common-support interval [0, u_P95].
  u_P95 is the 95th percentile of u pooled over all QC cities.
  The same 10-bin grid is applied to all distance coordinates.

  bin_edges  — (11,) array saved to outputs for downstream scripts
  radial_bin — integer 0–9 (0 = core, 9 = periphery); NaN if outside [0, u_P95]

Outputs
-------
  outputs/tables/radial_cells.parquet
      one row per cell, columns:
        city, country, label_k17, group4, Pop2022, cx, cy
        dist_m_primary, dist_km_primary, u_scaled, q_relative, d_km
        radial_bin          (main, scaled)
        radial_bin_q        (robustness, relative radius)
        radial_bin_d        (supplement, absolute km; bin edges = same quantiles)
        tier                (T1/T2/T3/T4, from city_tiers.parquet)

  outputs/tables/bin_grid.json
      {"n_bins":10, "u_edges":[...], "q_edges":[...], "d_edges":[...],
       "u_P95": ..., "alpha": ...}
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT        = Path(__file__).parent
PART4_DIR   = ROOT.parent / "core" / "outputs" / "tables"
OUT_TABLES  = ROOT / "outputs" / "tables"

CELL_PQ     = PART4_DIR / "cell_radial_position.parquet"
R98_PQ      = PART4_DIR / "city_R98.parquet"
PARAMS_JSON = PART4_DIR / "allometric_params.json"
TIERS_PQ    = OUT_TABLES / "city_tiers.parquet"

N_BINS      = 10
P_CUT       = 95    # common-support percentile for upper truncation


def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    # ── load params ──────────────────────────────────────────────────────
    log.info("Loading allometric_params.json ...")
    with open(PARAMS_JSON) as f:
        params = json.load(f)
    alpha = float(params["alpha_R98"])

    # ── load tier membership ─────────────────────────────────────────────
    log.info("Loading city_tiers.parquet ...")
    tiers = pd.read_parquet(TIERS_PQ, columns=["city", "country", "tier", "pop"])
    tier_map = tiers.set_index(["city", "country"])["tier"].to_dict()
    pop_map  = tiers.set_index(["city", "country"])["pop"].to_dict()

    # ── load R98 per city ────────────────────────────────────────────────
    log.info("Loading city_R98.parquet ...")
    r98 = pd.read_parquet(R98_PQ, columns=["city", "country", "R98_km"]) \
            .drop_duplicates(["city", "country"])
    r98["R98_m"] = r98["R98_km"] * 1000.0
    r98_map = r98.set_index(["city", "country"])["R98_m"].to_dict()

    # ── load cell data ───────────────────────────────────────────────────
    log.info("Loading cell_radial_position.parquet ...")
    needed = ["city", "country", "label_k17", "group4", "Pop2022",
              "cx", "cy", "dist_m_primary", "dist_km_primary",
              "dist_m_pop"]
    cells = pd.read_parquet(CELL_PQ, columns=needed)
    log.info(f"  {len(cells):,} cells loaded.")

    # ── filter to QC cities in tier table ───────────────────────────────
    cells["_key"] = list(zip(cells["city"], cells["country"]))
    valid_keys = set(tier_map.keys())
    cells = cells[cells["_key"].isin(valid_keys)].copy()
    log.info(f"  {len(cells):,} cells in QC cities.")

    # ── compute distance coordinates ─────────────────────────────────────
    log.info("Computing distance coordinates ...")
    cells["tier"] = cells["_key"].map(tier_map)
    cells["pop"]  = cells["_key"].map(pop_map)

    # scaled distance  u = dist_m / Pop^alpha
    cells["u_scaled"] = cells["dist_m_primary"] / (cells["pop"] ** alpha)

    # relative radius  q = dist_m / R98_m  (0–1 range within 98th pct)
    cells["R98_m"]    = cells["_key"].map(r98_map)
    cells["q_relative"] = cells["dist_m_primary"] / cells["R98_m"]

    # absolute km (already stored)
    cells["d_km"] = cells["dist_km_primary"]
    cells.drop(columns=["_key", "pop", "R98_m"], inplace=True)

    # ── common support: pool P0–P_CUT of u_scaled ────────────────────────
    log.info(f"Computing common-support interval [0, u_P{P_CUT}] for scaled distance ...")
    u_lo  = 0.0
    u_hi  = float(np.nanpercentile(cells["u_scaled"], P_CUT))
    log.info(f"  u range: [0, {u_hi:.2f}]")

    u_edges = np.linspace(u_lo, u_hi, N_BINS + 1)

    # for q and d: same P_CUT of the respective coordinate
    q_hi    = float(np.nanpercentile(cells["q_relative"], P_CUT))
    d_hi    = float(np.nanpercentile(cells["d_km"], P_CUT))
    q_edges = np.linspace(0, q_hi, N_BINS + 1)
    d_edges = np.linspace(0, d_hi, N_BINS + 1)

    # ── assign radial bins ────────────────────────────────────────────────
    log.info("Assigning radial bins ...")

    def _assign_bins(vals: pd.Series, edges: np.ndarray) -> pd.Series:
        """0-indexed bin; NaN for values outside [edges[0], edges[-1]]."""
        bins = pd.cut(vals, bins=edges, labels=False, include_lowest=True)
        return bins.astype("Int8")

    cells["radial_bin"]   = _assign_bins(cells["u_scaled"],   u_edges)
    cells["radial_bin_q"] = _assign_bins(cells["q_relative"], q_edges)
    cells["radial_bin_d"] = _assign_bins(cells["d_km"],       d_edges)

    n_valid = cells["radial_bin"].notna().sum()
    log.info(f"  {n_valid:,} / {len(cells):,} cells assigned to main radial bins "
             f"({100*n_valid/len(cells):.1f}%).")

    # ── save cells ────────────────────────────────────────────────────────
    keep = ["city", "country", "label_k17", "group4", "Pop2022",
            "cx", "cy", "dist_m_primary", "dist_km_primary",
            "dist_m_pop", "u_scaled", "q_relative", "d_km",
            "radial_bin", "radial_bin_q", "radial_bin_d", "tier"]
    _path = OUT_TABLES / "radial_cells.parquet"
    cells[keep].to_parquet(_path, index=False)
    _chk = pd.read_parquet(_path, columns=["city"])
    assert len(_chk) == len(cells), \
        f"Parquet read-back mismatch: wrote {len(cells)}, read {len(_chk)}"
    log.info(f"Saved + verified: radial_cells.parquet  ({len(cells):,} rows)")

    # ── save bin grid ─────────────────────────────────────────────────────
    grid = {
        "n_bins":   N_BINS,
        "p_cut":    P_CUT,
        "alpha":    alpha,
        "u_P95":    float(u_hi),
        "u_edges":  u_edges.tolist(),
        "q_P95":    float(q_hi),
        "q_edges":  q_edges.tolist(),
        "d_P95_km": float(d_hi),
        "d_edges":  d_edges.tolist(),
    }
    with open(OUT_TABLES / "bin_grid.json", "w") as f:
        json.dump(grid, f, indent=2)
    log.info("Saved: bin_grid.json")

    # ── summary ───────────────────────────────────────────────────────────
    print(f"\n=== Radial coordinate summary ===")
    print(f"  Common support (scaled):   u ∈ [0, {u_hi:.2f}] → {N_BINS} bins of "
          f"width {(u_hi-u_lo)/N_BINS:.2f}")
    print(f"  Common support (relative): q ∈ [0, {q_hi:.3f}] → {N_BINS} bins")
    print(f"  Common support (abs km):   d ∈ [0, {d_hi:.2f} km] → {N_BINS} bins")
    print(f"\n  Cells with valid main bin: {n_valid:,} / {len(cells):,}")
    print(f"\n  Bin centres (scaled distance):")
    centres = (u_edges[:-1] + u_edges[1:]) / 2
    for i, c in enumerate(centres):
        print(f"    bin {i:2d}: u = {c:.2f}")

    log.info("Done.")


if __name__ == "__main__":
    main()
