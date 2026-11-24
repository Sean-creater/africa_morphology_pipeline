"""
17_figure_main_embedding.py
=============================
Main 4-panel figure for the embedding-based center-periphery framework
(see extended/README.md for the full canonical pipeline).

Panels c/d are split by Africapolis size tier (not a tercile/quartile
split fit to this data) so that how population size reshapes the
core-periphery trajectory and composition profile is shown directly,
rather than through a regression-coefficient or bar-chart proxy. See
20_trajectory_by_africapolis_tier.py's docstring for the tier definition.

Layout (2 rows x 2 columns):
  a  Population centre + standardised radial coordinate (illustrative,
     one example city — embeds 11_accra_rings_viz.py's rendered
     accra_rings.png)
  b  Built-up coverage vs. comprehensive morphology: out-of-sample R^2
     for predicting cell population density, and the density
     fold-difference at matched coverage (14_coverage_vs_morphology.py)
  c  Core-periphery morphological trajectory D(q) in embedding space,
     one line + fitted D_max*(1-exp(-q/tau)) curve per Africapolis size
     tier (20_trajectory_by_africapolis_tier.py) — shows directly how
     city size reshapes the transition (tau/D_max both increase with
     size), replacing the old pooled single-line + twin-axis dispersion
     version.
  d  4-group composition profile along the radius, one stacked-area
     sub-panel per Africapolis size tier, same q-axis — shows directly
     how the core-to-periphery compositional shift differs by city size,
     replacing the old pooled 17-type+4-group pair.

Each data-loading call degrades gracefully (logs a warning and leaves
that panel blank) if its upstream table isn't available yet.

Outputs
-------
  outputs/figures/<YYYY-MM-DD>/figure_main_embedding.svg / .png

Usage
-----
    python 17_figure_main_embedding.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.optimize import curve_fit
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
    "font.family":  "Arial",
    "font.size":    8,
    "svg.fonttype": "none",
    "mathtext.fontset": "custom",
    "mathtext.rm":       "Arial",
    "mathtext.it":       "Arial:italic",
    "mathtext.bf":       "Arial:bold",
})

ROOT       = Path(__file__).parent
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIG    = ROOT / "outputs" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

# Pre-rendered illustration (Accra example city), produced by
# 11_accra_rings_viz.py using the q = d/R98 radial definition and the
# 3-4-3 Core/Mid/Outer zone split.
ACCRA_PNG = OUT_FIG / "accra_rings.png"

EMB_COLOR = "#293890"

# Same 4-group colours as 02_plot_hierarchy.py / cluster_profiles_heatmap.svg's
# GROUP4_COLORS.
PALETTE_4 = {1: "#8DA0CB", 2: "#FC8D62", 3: "#66C2A5", 4: "#E78AC3"}
GROUP_NAMES = {
    1: "Cropland-Fringe\nLow-Rise",
    2: "Heterogeneous\nSparse Fabric",
    3: "Large-Volume\nCompact Fabric",
    4: "Dense Small-Building\nFabric",
}

# Africapolis (OECD/SWAC) size tiers — see 20_trajectory_by_africapolis_tier.py
TIER_LABELS = ["Small (<100k)", "Intermediate (100k-1M)", "Large (>=1M)"]
TIER_COLORS = {"Small (<100k)": "#8DA0CB",
              "Intermediate (100k-1M)": "#FFD700",
              "Large (>=1M)": "#C8292A"}
TIER_SHORT  = {"Small (<100k)": "Small", "Intermediate (100k-1M)": "Intermediate",
              "Large (>=1M)": "Large"}


def _load(fname):
    p = OUT_TABLES / fname
    if not p.exists():
        log.warning("Missing (panel will be blank): %s", p)
        return None
    return pd.read_csv(p)


def _style_ax(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#AAAAAA")
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(colors="#555555", labelsize=6.5)


# ---------------------------------------------------------------------------
# Panel b: coverage vs. morphology
# ---------------------------------------------------------------------------
def draw_panel_b(ax, summary):
    if summary is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    s = summary.set_index("statistic")["value"]
    r2_m1 = s.get("r2_oos_M1_coverage", np.nan)
    r2_m2 = s.get("r2_oos_M2_coverage_embedding", np.nan)
    ratio = s.get("density_ratio_p90_p10_at_matched_coverage", np.nan)

    ax.bar([0, 1], [r2_m1, r2_m2], color=["#999999", EMB_COLOR], width=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Coverage\nonly", "+ full\nmorphology"], fontsize=6.5)
    ax.set_ylabel("Out-of-sample $R^2$\n(cell population density)", fontsize=7)
    for x, v in zip([0, 1], [r2_m1, r2_m2]):
        ax.text(x, v + 0.015, f"{v:.3f}", ha="center", fontsize=6.5)
    # Callout box for the density-ratio figure, set apart from the R^2
    # bars since it is the panel's primary readout. Cross-validated
    # (held-out) association, not a causal effect estimate.
    if pd.notna(ratio):
        ax.text(0.03, 0.97, f"{ratio:.1f}$\\times$",
               transform=ax.transAxes, fontsize=13, fontweight="bold",
               ha="left", va="top", color=EMB_COLOR)
        ax.text(0.03, 0.82, "density difference at\nmatched built-up coverage\n(90th vs 10th pctile.)",
               transform=ax.transAxes, fontsize=5.8, ha="left", va="top",
               color="#333333")
    ax.set_ylim(0, max(r2_m1, r2_m2) * 1.35 if pd.notna(r2_m2) else 1)
    _style_ax(ax)
    ax.set_title("b", loc="left", fontsize=9, fontweight="bold")


# ---------------------------------------------------------------------------
# Panel c: D(q) trajectory by Africapolis size tier, each with its own
# fitted saturation curve
# ---------------------------------------------------------------------------
def _exp_saturation(q, D_max, tau):
    return D_max * (1.0 - np.exp(-q / tau))


def draw_panel_c(ax, traj_tier, fits_tier):
    if traj_tier is None or fits_tier is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    fits = fits_tier.set_index("tier")
    for tier in TIER_LABELS:
        sub = traj_tier[traj_tier["tier"] == tier].dropna(subset=["D_q"])
        if len(sub) == 0:
            continue
        color = TIER_COLORS[tier]
        ax.plot(sub["q_mid"], sub["D_q"], color=color, lw=0, marker="o",
               markersize=2.5, zorder=3)
        if tier in fits.index and np.isfinite(fits.loc[tier, "tau"]):
            D_max, tau = fits.loc[tier, "D_max"], fits.loc[tier, "tau"]
            q_grid = np.linspace(0, sub["q_mid"].max(), 200)
            ax.plot(q_grid, _exp_saturation(q_grid, D_max, tau),
                   color=color, lw=1.6, zorder=2,
                   label=f"{TIER_SHORT[tier]} ($\\tau$={tau:.2f})")

    ax.set_xlabel("Relative radius $q = d / R_{98}$", fontsize=7)
    ax.set_ylabel("Embedding distance\nfrom own core, $D(q)$", fontsize=7)
    ax.legend(fontsize=6, frameon=False, loc="upper left", title="City size",
             title_fontsize=6)
    _style_ax(ax)
    ax.set_title("c", loc="left", fontsize=9, fontweight="bold")


# ---------------------------------------------------------------------------
# Panel d: 4-group composition profile, one sub-panel per Africapolis tier
# ---------------------------------------------------------------------------
def draw_panel_d(fig, gs_cell, group_tier):
    gs_d = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_cell, wspace=0.08)
    axes_d = [fig.add_subplot(gs_d[0, i]) for i in range(3)]

    if group_tier is None:
        axes_d[0].text(0.5, 0.5, "No data", ha="center", va="center",
                       transform=axes_d[0].transAxes)
        return axes_d

    for i, tier in enumerate(TIER_LABELS):
        ax = axes_d[i]
        sub = group_tier[group_tier["tier"] == tier]
        if len(sub) == 0:
            continue
        q = sub["q_mid"].values
        shares = np.vstack([sub[f"g_{k}"].values for k in range(1, 5)])
        colors4 = [PALETTE_4[k] for k in range(1, 5)]
        ax.stackplot(q, shares, colors=colors4, linewidth=0)
        ax.set_xlim(q.min(), q.max())
        ax.set_ylim(0, 1)
        ax.set_title(TIER_SHORT[tier], fontsize=6.5, color=TIER_COLORS[tier],
                    fontweight="bold")
        ax.set_xticks([0, 0.5])
        ax.tick_params(labelsize=6)
        if i == 1:
            ax.set_xlabel("Relative radius $q$", fontsize=7)
        if i == 0:
            ax.set_ylabel("Composition share", fontsize=7)
        else:
            ax.tick_params(labelleft=False)
        _style_ax(ax)

    axes_d[0].set_title("d", loc="left", fontsize=9, fontweight="bold")
    axes_d[-1].legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=PALETTE_4[k]) for k in range(1, 5)],
        labels=[GROUP_NAMES[k] for k in range(1, 5)],
        fontsize=5, frameon=False, loc="upper left", bbox_to_anchor=(1.03, 1.0))
    return axes_d


def main():
    log.info("Loading panel data ...")
    cov_summary = _load("coverage_vs_morphology_summary.csv")
    traj_tier   = _load("trajectory_by_africapolis_tier.csv")
    fits_tier   = _load("trajectory_curve_fits_africapolis.csv")
    group_tier  = _load("group_profile_by_africapolis_tier.csv")

    fig = plt.figure(figsize=(175 / 25.4, 110 / 25.4))
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1.15])

    ax_a = fig.add_subplot(gs[0, 0])
    if ACCRA_PNG.exists():
        img = mpimg.imread(ACCRA_PNG)
        ax_a.imshow(img)
        ax_a.axis("off")
    else:
        log.warning("Missing (panel a will be blank): %s", ACCRA_PNG)
        ax_a.text(0.5, 0.5, "a: population centre +\nradial coordinate\n"
                  "(see accra_rings.svg)",
                  ha="center", va="center", transform=ax_a.transAxes,
                  fontsize=7, color="#555555")
        ax_a.axis("off")
    ax_a.set_title("a", loc="left", fontsize=9, fontweight="bold")

    draw_panel_b(fig.add_subplot(gs[0, 1]), cov_summary)
    draw_panel_c(fig.add_subplot(gs[1, 0]), traj_tier, fits_tier)
    draw_panel_d(fig, gs[1, 1], group_tier)

    fig.tight_layout(pad=1.2)
    fig.text(0.5, -0.02,
            "a: Accra is one illustrative example city, not part of the method "
            "definition; the centre is the population-weighted primary cluster "
            r"centroid, $R_{98}$ is that city's own 98th-percentile cell distance. "
            "b: the density ratio is a cross-validated (held-out) association at "
            "matched coverage, not a causal effect estimate.",
            fontsize=5.5, color="#555555", style="italic", ha="center", va="top")
    for ext in ("svg", "png"):
        out = OUT_FIG / f"figure_main_embedding.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        log.info("Saved: %s", out)
    plt.close(fig)

    log.info("Done.")


if __name__ == "__main__":
    main()
