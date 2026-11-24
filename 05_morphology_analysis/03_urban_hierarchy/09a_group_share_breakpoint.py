"""
09a_group_share_breakpoint.py
===============================
Identifies the population range where each of the 4 morphology groups'
composition share shifts fastest, computed directly on
02_plot_hierarchy.py's p1-p4 curves (4-group composition SHARE vs.
log(population)), rather than on H17_star/r17 (diversity entropy /
spatial assortativity), which are scalar summaries one step removed from
composition itself.

Method: refit the same LOESS curves 02_plot_hierarchy.py already plots
(same x_grid, same LOESS_FRAC=0.35 — not re-deriving anything, just the
point-estimate curve, no bootstrap needed here) for all 4 groups, then
for each group curve compute the local slope dp_g/d(log_pop) via finite
differences and find where |slope| changes fastest (max |second
derivative| = point of sharpest curvature = the group's own "breakpoint"
on the composition curve itself). Compare where these four per-group
curvature peaks cluster.

This directly answers "does the SHARE of each of the 4 morphology groups
bend at a common city size" instead of asking it indirectly through H17
(entropy of the shares) or r17 (spatial arrangement of the groups).

Output
------
  outputs/tables/group_share_breakpoint.csv
      group | pop_at_max_curvature | log_pop_at_max_curvature
  Prints the per-group curvature peak and the common range across groups.

Usage
-----
    python 09a_group_share_breakpoint.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT    = Path(__file__).parent
METRICS = ROOT / "outputs" / "tables" / "city_metrics.parquet"
OUT_TABLES = ROOT / "outputs" / "tables"

LOESS_FRAC = 0.35   # same as 02_plot_hierarchy.py
GROUP_NAMES = {
    1: "Cropland-Fringe Low-Rise",
    2: "Heterogeneous Sparse Fabric",
    3: "Large-Volume Compact Fabric",
    4: "Dense Small-Building Fabric",
}


def _fit_loess(x, y, x_grid, frac=LOESS_FRAC):
    mask = np.isfinite(x) & np.isfinite(y)
    sm = lowess(y[mask], x[mask], frac=frac, return_sorted=True)
    return np.interp(x_grid, sm[:, 0], sm[:, 1])


def main():
    log.info("Loading city_metrics.parquet ...")
    df = pd.read_parquet(METRICS, columns=["log_pop", "p1", "p2", "p3", "p4"])
    df = df.dropna(subset=["log_pop", "p1", "p2", "p3", "p4"]).copy()
    log.info("  %d cities.", len(df))

    # Same x_grid construction as 02_plot_hierarchy.py (1st-99.5th pctile)
    x_lo = np.nanpercentile(df["log_pop"], 1)
    x_hi = np.nanpercentile(df["log_pop"], 99.5)
    x_grid = np.linspace(x_lo, x_hi, 400)

    curves = {g: _fit_loess(df["log_pop"].values, df[f"p{g}"].values, x_grid)
              for g in range(1, 5)}
    pop_grid = np.exp(x_grid)

    # Single-point "max curvature" is noisy on a LOESS curve (second
    # derivative amplifies local wiggle). More robust: for each group,
    # find the SHORTEST log_pop window that contains a fixed fraction
    # (here 50%) of that group's total curve movement (|p_g(x_hi) -
    # p_g(x_lo)|) — i.e. where the change is most CONCENTRATED, not just
    # where the instantaneous second derivative peaks.
    rows = []
    windows = {}
    for g in range(1, 5):
        y = curves[g]
        total_change = y[-1] - y[0]
        cum_change = (y - y[0]) / total_change if total_change != 0 else np.zeros_like(y)
        # window containing the steepest 50% of cumulative change
        lo_idx = np.argmin(np.abs(cum_change - 0.25))
        hi_idx = np.argmin(np.abs(cum_change - 0.75))
        lo_idx, hi_idx = sorted([lo_idx, hi_idx])
        windows[g] = (pop_grid[lo_idx], pop_grid[hi_idx])
        rows.append({
            "group": g, "group_name": GROUP_NAMES[g],
            "total_change": total_change,
            "pop_25pct_change": pop_grid[lo_idx],
            "pop_75pct_change": pop_grid[hi_idx],
        })
        log.info("  Group %d (%s): middle 50%% of change occurs between pop %s and %s",
                 g, GROUP_NAMES[g], f"{pop_grid[lo_idx]:,.0f}", f"{pop_grid[hi_idx]:,.0f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "group_share_breakpoint.csv", index=False, float_format="%.4f")
    log.info("Saved: group_share_breakpoint.csv")

    print("\n=== Per-group composition-share curve: where the middle 50% of change happens ===")
    print(out.to_string(index=False))

    # Groups 1 and 4 move the most (see checkpoint table) — they're the
    # ones that actually define where a "small/large" split would matter.
    big_movers = [1, 4]
    los = [windows[g][0] for g in big_movers]
    his = [windows[g][1] for g in big_movers]
    print(f"\nGroups 1+4 (the two that move the most): transition window "
         f"{min(los):,.0f} - {max(his):,.0f}")
    print(f"Overlap (both groups' middle-50%% windows): "
         f"{max(los):,.0f} - {min(his):,.0f}")

    log.info("Done.")


if __name__ == "__main__":
    main()
