"""
25_trajectory_by_tier_combined.py
====================================
Parallel companion to 21_trajectory_by_community.py's single-panel
D(q) summary figure, but stratified by Africapolis (OECD/SWAC) city
SIZE tier (Small/Intermediate/Large) instead of regional morphology
community. Answers "how does city size reshape the core-periphery
trajectory" as its own standalone figure, at the same visual weight as
the community version -- previously this was only a cramped 3-line
sub-panel (panel c) inside 17_figure_main_embedding.py's 4-panel main
figure, with no CI bands or direct D(q_max) labels, so the Large tier's
wide bootstrap uncertainty (n=88 cities, tau 95% CI [0.44, 1.18]) was
invisible.

Pure plotting script -- no new computation. Reuses:
  - trajectory_by_africapolis_tier.csv (20_trajectory_by_africapolis_
    tier.py's city-weighted, bootstrap-CI, origin-anchored D(q) fits)
  - trajectory_by_community.csv's community==-1 rows for the pooled
    (all-cities) reference curve -- identical pooled definition as
    21_trajectory_by_community.py, so the two figures' dashed reference
    lines are directly comparable
  - trajectory_curve_fits_africapolis.csv for D_max/tau and their CIs

Design identical to 21's redesigned single panel (per methods review):
curves + pooled dashed reference + CI bands (drawn on observed bins,
not interpolated onto the smooth fit grid) + end-of-curve D(q_max)
labels, pooled line on top (highest zorder). No scatter points, no
second parameter panel -- same rationale as 21: the per-tier
differences are already legible from curve shape, and D_max/tau/R2 with
full CIs are reported in trajectory_curve_fits_africapolis.csv.

Output
------
  outputs/figures/trajectory_by_tier_combined.svg/.png

Usage
-----
    python 25_trajectory_by_tier_combined.py
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
BIN_GRID   = OUT_TABLES / "bin_grid.json"
OUT_FIG    = ROOT / "outputs" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

POOLED_COLOR = "#555555"
TIER_LABELS = ["Small (<100k)", "Intermediate (100k-1M)", "Large (>=1M)"]
TIER_COLORS = {"Small (<100k)": "#8DA0CB",
              "Intermediate (100k-1M)": "#FFD700",
              "Large (>=1M)": "#C8292A"}
TIER_SHORT  = {"Small (<100k)": "Small", "Intermediate (100k-1M)": "Intermediate",
              "Large (>=1M)": "Large"}


def _exp_saturation(q, D_max, tau):
    return D_max * (1.0 - np.exp(-q / tau))


def main():
    log.info("Loading trajectory_by_africapolis_tier.csv + trajectory_curve_fits_africapolis.csv ...")
    tier_traj = pd.read_csv(OUT_TABLES / "trajectory_by_africapolis_tier.csv")
    tier_fit = pd.read_csv(OUT_TABLES / "trajectory_curve_fits_africapolis.csv")

    log.info("Loading pooled reference from trajectory_by_community.csv (community==-1) ...")
    comm_traj = pd.read_csv(OUT_TABLES / "trajectory_by_community.csv")
    comm_fit = pd.read_csv(OUT_TABLES / "trajectory_curve_fits_by_community.csv")
    pooled_traj = comm_traj[comm_traj["community"] == -1].dropna(subset=["D_q"])
    pooled_fit = comm_fit[comm_fit["community"] == -1].iloc[0]

    with open(BIN_GRID) as f:
        q_max = float(json.load(f)["q_P95"])

    # Curve/CI bands terminate at the last bin with data (same fix as
    # 21_trajectory_by_community.py -- avoids the fitted curve visibly
    # overshooting past where its own CI band ends, since a bin's centre
    # sits half a bin-width short of q_max).
    q_obs_end = float(pooled_traj["q_mid"].iloc[-1])
    q_grid = np.linspace(0, q_obs_end, 200)

    log.info("Drawing single-panel tier trajectory summary figure ...")
    # Fixed absolute axes box (mm), identical to
    # 21_trajectory_by_community.py's single panel AND
    # 23_scale_dependent_convergence.py's panel a (100x75mm, same
    # margins) -- so all three figures' plot areas align exactly when
    # stacked. Font sizes (9pt axis labels, 8pt ticks), axis colour
    # (black), and spine linewidth (0.8) are also unified across all
    # three.
    AXES_W_MM, AXES_H_MM = 100.0, 75.0
    LEFT_MM, RIGHT_MM, BOTTOM_MM, TOP_MM = 20.0, 60.0, 16.0, 6.0
    FIG_W_MM = LEFT_MM + AXES_W_MM + RIGHT_MM
    FIG_H_MM = BOTTOM_MM + AXES_H_MM + TOP_MM
    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
    ax = fig.add_axes([LEFT_MM / FIG_W_MM, BOTTOM_MM / FIG_H_MM,
                       AXES_W_MM / FIG_W_MM, AXES_H_MM / FIG_H_MM])

    D_at_qmax_pooled = _exp_saturation(q_max, pooled_fit["D_max"], pooled_fit["tau"])
    ax.fill_between(pooled_traj["q_mid"], pooled_traj["D_q_lo"], pooled_traj["D_q_hi"],
                    color=POOLED_COLOR, alpha=0.18, linewidth=0, zorder=1)

    end_labels = []
    for tier in TIER_LABELS:
        color = TIER_COLORS[tier]
        fit_row = tier_fit[tier_fit["tier"] == tier].iloc[0]
        if not np.isfinite(fit_row["tau"]):
            continue
        tier_sub = tier_traj[tier_traj["tier"] == tier].dropna(subset=["D_q"])
        ax.fill_between(tier_sub["q_mid"], tier_sub["D_q_lo"], tier_sub["D_q_hi"],
                        color=color, alpha=0.10, linewidth=0, zorder=3)
        y_curve = _exp_saturation(q_grid, fit_row["D_max"], fit_row["tau"])
        D_at_qmax = _exp_saturation(q_max, fit_row["D_max"], fit_row["tau"])
        ax.plot(q_grid, y_curve, color=color, lw=1.8, zorder=4)
        end_labels.append((tier, color, y_curve[-1], D_at_qmax,
                          int(fit_row["n_cities"])))

    # De-collide end-of-curve labels.
    end_labels.sort(key=lambda t: t[2])
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0] or 1.0
    min_gap = 0.07 * y_range
    placed_y = []
    for tier, color, y_end, D_at_qmax, n_cities in end_labels:
        y_label = y_end if not placed_y else max(y_end, placed_y[-1] + min_gap)
        placed_y.append(y_label)
        ax.text(q_grid[-1] * 1.02, y_label,
               f"{TIER_SHORT[tier]} (n={n_cities})  $D$({q_max:.1f})={D_at_qmax:.2f}",
               color=color, fontsize=7, fontweight="bold",
               va="center", ha="left", clip_on=False)

    # Pooled line drawn LAST, on top of every tier curve/CI band.
    ax.plot(q_grid, _exp_saturation(q_grid, pooled_fit["D_max"], pooled_fit["tau"]),
           color=POOLED_COLOR, lw=2.8, ls="--", zorder=10,
           label=f"Pooled, $D$({q_max:.1f})={D_at_qmax_pooled:.2f}")

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

    # NOTE: no fig.tight_layout() here -- axes box fixed in absolute mm
    # via add_axes() above, matching 21/23's convention.
    caption = (
        r"City size tiers: Africapolis (OECD/SWAC) classes -- Small <100k, "
        "Intermediate 100k-1M, Large" "\n"
        r">=1M population. $D(q_{max})$ is the fitted $D(q)=D_{max}(1-e^{-q/\tau})$ "
        f"curve evaluated at $q_{{max}}$={q_max:.2f}" "\n"
        "(the common-support cutoff) -- a direct read within the data's support, not the "
        r"extrapolated $D_{max}$." "\n"
        r"Full $D_{max}$, $\tau$ (characteristic radius -- not an inflection point) and "
        "their 95% country-block" "\n"
        "bootstrap CIs are reported in trajectory_curve_fits_africapolis.csv "
        "(Large tier: n=88 cities,"  "\n"
        r"$\tau$ 95% CI is wide -- interpret with caution)."
    )
    fig.text(0.5, -0.02, caption, fontsize=5.5, color="#555555", style="italic",
            ha="center", va="top", linespacing=1.6)

    for ext in ("svg", "png"):
        out = OUT_FIG / f"trajectory_by_tier_combined.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        log.info("Saved: %s", out)
    plt.close(fig)

    log.info("Done.")


if __name__ == "__main__":
    main()
