"""
08_robustness.py
================
Robustness checks for the city-size scaling regressions (06_scaling_regression.py).

Seven specifications per outcome (six for r17 — see Rarefied, below), all
standardised OLS with country-clustered SE:

  Main          : Y_z ~ X_z
  Community FE  : Y_z ~ X_z + C(community)
  Country FE    : Y_z ~ X_z + C(country)
  ≥50 cells     : Y_z ~ X_z  [cities with n_cells ≥ 50 only]
  4-group       : H_star / r  [4-group versions of the main outcomes]
  Size control  : Y_z ~ X_z + log(size)  [size = n_cells for H17_star,
                  n_edges for r17 — both outcomes are mechanically bounded
                  by their own "sample size" (see 06_scaling_regression.py
                  docstring); this checks the slope survives controlling
                  for it directly, as a complement to the ≥50-cells filter
                  above, which only excludes small cities rather than
                  equalising sampling effort across the kept ones]
  Rarefied      : H17_star only. Y_z ~ X_z where Y = H17_star_rare
                  (H17_star recomputed on a fixed 50-cell random subsample
                  per city, averaged over 50 draws — see
                  01_compute_city_metrics.py). Unlike "Size control" above
                  (which adjusts for n_cells as a covariate but still
                  computes H17_star on each city's full, unequal cell
                  count) or "≥50 cells" (which filters but doesn't
                  equalise), this evaluates every included city at
                  identical sampling effort, the most direct test of
                  whether the population-diversity slope is a genuine
                  diversity effect vs. an artefact of bigger cities simply
                  having more grid cells.

Outcomes: H17_star, r17 and r17_z (main); H_star and r (4-group
robustness, H17_star/r17 only — no 4-group r17_z computed).

r17_z is a permutation-null z-score (computed in
01_compute_city_metrics.py): hold each city's own Queen graph AND its
17-type count distribution fixed, permute which node gets which type
label (N_PERM=200 draws), and score the observed r17 against that null.
This supersedes the plain "+log(n_edges)" control as the primary test of
whether r17's population relationship is genuine: n_edges is a composite
of cell count, graph topology AND population-driven spatial extent, so
controlling for it as a covariate is a conditional association, not
confound removal, and its negative coefficient should not be read as
proof the raw positive r17-population relationship is an artefact. r17_z
nets out graph size/topology/composition at the single-city level before
any regression, and its own population relationship is POSITIVE and
survives both community FE and a residual size check — the opposite
conclusion from the naive n_edges-covariate version. r17_z gets its own
"Size control" row (as a residual-dependence check, not the primary
test) but no "4-group" or "Rarefied" rows (no 4-group z-score computed;
rarefying a spatial adjacency graph changes its topology in a way that
isn't a simple fixed-size subsample the way entropy's is).

Output
------
    outputs/tables/robustness.csv
    outputs/figures/robustness_coef.pdf/png

Usage
-----
    python 08_robustness.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# Register a font whose internal name is "Arial" so matplotlib's mathtext
# engine (which resolves mathtext.rm/it/bf by actually looking up a font,
# unlike plain text under svg.fonttype=none) can find one — otherwise it
# silently falls back to DejaVu Sans for equation glyphs in the SVG output.
_FONT_DIR = Path("/home/sjs96_file_share/Scripts4Analysis/_fonts")
if _FONT_DIR.is_dir():
    for _f in fm.findSystemFonts(fontpaths=[str(_FONT_DIR)]):
        fm.fontManager.addfont(_f)
else:
    logging.warning("Font directory not found (%s); falling back to default sans-serif.", _FONT_DIR)

matplotlib.rcParams.update({
    "font.family":     "Arial",
    "font.size":       8,
    "svg.fonttype": "none",
    "mathtext.fontset": "custom",
    "mathtext.rm":       "Arial",
    "mathtext.it":       "Arial:italic",
    "mathtext.bf":       "Arial:bold",
})

ROOT    = Path(__file__).resolve().parent
METRICS = ROOT / "outputs" / "tables" / "city_metrics.parquet"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4

MIN_CELLS = 50   # threshold for sample-sensitivity check
N_RAREFY  = 50   # must match 01_compute_city_metrics.py's N_RAREFY

# Mechanical "sample size" each outcome is bounded by — see
# 06_scaling_regression.py's docstring for the full rationale.
SIZE_COVARIATE = {
    "H17_star": "n_cells",
    "r17":      "n_edges",
    "r17_z":    "n_edges",
}

# Specification display labels (ordered for plot)
SPEC_LABELS = [
    "Main",
    "+ Community FE",
    "+ Country FE",
    r"Cities $\geq$50 cells",
    "4-group metrics",
    "+ log(size)",
    f"Rarefied (n={N_RAREFY})",
]

OUTCOME_COLORS = {
    "H17_star": "#2166AC",
    "r17":      "#D6604D",
    "r17_z":    "#8B4A9C",
}

OUTCOME_LABELS = {
    "H17_star": r"$H^*$ (diversity)",
    "r17":      r"$r$ (assortativity)",
    "r17_z":    r"$r$ permutation-null $z$",
}


# ---------------------------------------------------------------------------
# OLS helper
# ---------------------------------------------------------------------------
def run_ols(df, formula, cluster_var="country"):
    return smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster_var]},
    )


def extract_coef(result, param="X_z"):
    ci = result.conf_int().loc[param]
    return dict(
        beta   = result.params[param],
        ci_lo  = ci[0],
        ci_hi  = ci[1],
        pval   = result.pvalues[param],
        r2     = result.rsquared,
        n      = int(result.nobs),
    )


def standardise(df, x_col, y_col):
    """Add X_z and Y_z columns (in-place copy)."""
    d = df.copy()
    d["X_z"] = (d[x_col] - d[x_col].mean()) / d[x_col].std()
    d["Y_z"] = (d[y_col] - d[y_col].mean()) / d[y_col].std()
    return d


# ---------------------------------------------------------------------------
# Run all specifications
# ---------------------------------------------------------------------------
def run_robustness(df):
    records = []

    for outcome in ["H17_star", "r17", "r17_z"]:
        # Determine 4-group analogue (none for r17_z — no 4-group
        # permutation-null z-score computed)
        alt_outcome = {"H17_star": "H_star", "r17": "r"}.get(outcome)

        df_base = df.dropna(subset=[outcome, "log_pop", "community", "country"]).copy()

        logging.info("=== %s (n=%d) ===", outcome, len(df_base))

        # 1. Main
        d = standardise(df_base, "log_pop", outcome)
        c = extract_coef(run_ols(d, "Y_z ~ X_z"))
        c.update(outcome=outcome, spec="Main")
        records.append(c)
        logging.info("  Main:          β=%.4f [%.4f, %.4f]  R²=%.3f",
                     c["beta"], c["ci_lo"], c["ci_hi"], c["r2"])

        # 2. Community FE
        d = standardise(df_base, "log_pop", outcome)
        c = extract_coef(run_ols(d, "Y_z ~ X_z + C(community)"))
        c.update(outcome=outcome, spec="Community FE")
        records.append(c)
        logging.info("  + Community FE: β=%.4f [%.4f, %.4f]  R²=%.3f",
                     c["beta"], c["ci_lo"], c["ci_hi"], c["r2"])

        # 3. Country FE
        d = standardise(df_base, "log_pop", outcome)
        c = extract_coef(run_ols(d, "Y_z ~ X_z + C(country)"))
        c.update(outcome=outcome, spec="Country FE")
        records.append(c)
        logging.info("  + Country FE:   β=%.4f [%.4f, %.4f]  R²=%.3f",
                     c["beta"], c["ci_lo"], c["ci_hi"], c["r2"])

        # 4. Sample sensitivity: n_cells >= MIN_CELLS
        df_large = df_base[df_base["n_cells"] >= MIN_CELLS].copy()
        d = standardise(df_large, "log_pop", outcome)
        c = extract_coef(run_ols(d, "Y_z ~ X_z"))
        c.update(outcome=outcome, spec=f"Cities ≥{MIN_CELLS} cells")
        records.append(c)
        logging.info("  ≥%d cells (n=%d): β=%.4f [%.4f, %.4f]  R²=%.3f",
                     MIN_CELLS, len(df_large),
                     c["beta"], c["ci_lo"], c["ci_hi"], c["r2"])

        # 5. 4-group resolution (no analogue for r17_z — see above)
        if alt_outcome is not None:
            df_alt = df.dropna(subset=[alt_outcome, "log_pop", "community", "country"]).copy()
            d = standardise(df_alt, "log_pop", alt_outcome)
            c = extract_coef(run_ols(d, "Y_z ~ X_z"))
            c.update(outcome=outcome, spec="4-group")
            records.append(c)
            logging.info("  4-group (%s, n=%d): β=%.4f [%.4f, %.4f]  R²=%.3f",
                         alt_outcome, len(df_alt),
                         c["beta"], c["ci_lo"], c["ci_hi"], c["r2"])

        # 6. Size control — the outcome's own mechanical sample size
        # (n_cells for H17_star, n_edges for r17) as a covariate.
        size_col = SIZE_COVARIATE[outcome]
        df_size = df_base[df_base[size_col] > 0].copy()
        df_size["log_size"] = np.log(df_size[size_col])
        d = standardise(df_size, "log_pop", outcome)
        d["log_size"] = df_size["log_size"]
        c = extract_coef(run_ols(d, "Y_z ~ X_z + log_size"))
        c.update(outcome=outcome, spec="Size control")
        records.append(c)
        logging.info("  + log(%s): β=%.4f [%.4f, %.4f]  R²=%.3f",
                     size_col, c["beta"], c["ci_lo"], c["ci_hi"], c["r2"])

        # 7. Rarefied — H17_star only (see module docstring for why r17
        # has no analogue here).
        if outcome == "H17_star":
            df_rare = df.dropna(
                subset=["H17_star_rare", "log_pop", "community", "country"]
            ).copy()
            d = standardise(df_rare, "log_pop", "H17_star_rare")
            c = extract_coef(run_ols(d, "Y_z ~ X_z"))
            c.update(outcome=outcome, spec="Rarefied")
            records.append(c)
            logging.info("  Rarefied (n=%d, n_cities=%d): β=%.4f [%.4f, %.4f]  R²=%.3f",
                         N_RAREFY, len(df_rare),
                         c["beta"], c["ci_lo"], c["ci_hi"], c["r2"])

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Coefficient plot
# ---------------------------------------------------------------------------
def plot_robustness(rob_df):
    """
    One panel per outcome (H17_star | r17 | r17_z). Rows = up to 7
    specifications (fewer for r17/r17_z, which lack some rows — see
    module docstring); points = β; horizontal bars = 95% CI.
    """
    spec_order = ["Main", "Community FE", "Country FE",
                  f"Cities ≥{MIN_CELLS} cells", "4-group",
                  "Size control", "Rarefied"]

    outcomes = ["H17_star", "r17", "r17_z"]
    fig, axes = plt.subplots(1, len(outcomes),
                             figsize=(75 * len(outcomes) * MM, 90 * MM),
                             sharey=True)
    fig.patch.set_facecolor("white")

    for ax, outcome in zip(axes, outcomes):
        col = OUTCOME_COLORS[outcome]
        sub = rob_df[rob_df["outcome"] == outcome].copy()
        sub["spec_order"] = sub["spec"].map({s: i for i, s in enumerate(spec_order)})
        sub = sub.sort_values("spec_order")

        # Use each spec's fixed position in spec_order directly as its y
        # coordinate (not np.arange(len(sub))) so both panels share the
        # same row layout even though r17 has no "Rarefied" row — that
        # row is simply left blank in the right panel rather than
        # compacting the remaining rows and breaking row alignment
        # between the two outcomes.
        y_pos = sub["spec_order"].values
        ax.errorbar(
            sub["beta"], y_pos,
            xerr=[sub["beta"] - sub["ci_lo"], sub["ci_hi"] - sub["beta"]],
            fmt="o", color=col, ecolor=col, elinewidth=1.2,
            markersize=5, capsize=3, zorder=3,
        )

        # Annotate β
        for _, row in sub.iterrows():
            i = int(row["spec_order"])
            ax.text(row["ci_hi"] + 0.005, i,
                    f"β={row['beta']:.3f}",
                    va="center", ha="left", fontsize=6, color="#444444")

        ax.axvline(0, color="#AAAAAA", lw=0.8, ls="--", zorder=1)
        ax.set_yticks(range(len(spec_order)))

        # Y-tick labels only on left panel
        if ax is axes[0]:
            labels = []
            for s in spec_order:
                if s == f"Cities ≥{MIN_CELLS} cells":
                    labels.append(f"Cities ≥{MIN_CELLS} cells")
                elif s == "4-group":
                    labels.append("4-group metrics")
                elif s == "Rarefied":
                    labels.append(f"Rarefied (n={N_RAREFY})")
                else:
                    labels.append(s)
            ax.set_yticklabels(labels, fontsize=7)
        else:
            # NOT set_yticklabels([]) here: with sharey=True the two axes
            # share the same Y Axis label state, so setting axes[1]'s
            # labels to an explicit empty list wipes out the labels just
            # set on axes[0] too (both this and the left panel would end
            # up with blank y-ticks). Hiding the tick LABELS visually via
            # tick_params leaves the underlying (shared) text alone.
            ax.tick_params(labelleft=False)

        ax.set_xlabel("Standardised β (95% CI)", fontsize=7)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=8, fontweight="bold", pad=5)
        ax.invert_yaxis()

        ax.set_facecolor("white")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color("#AAAAAA")
            ax.spines[sp].set_linewidth(0.6)
        ax.tick_params(colors="#555555", labelsize=7)
        ax.xaxis.label.set_color("#555555")

    fig.tight_layout(pad=1.5, w_pad=2.0)
    for ext in ("svg", "png"):
        fig.savefig(FIG_DIR / f"robustness_coef.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Saved: robustness_coef.pdf/png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logging.info("Loading city metrics ...")
    df = pd.read_parquet(METRICS)
    df["community"] = pd.to_numeric(df["community"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["community"]).copy()
    df["community"] = df["community"].astype(int)
    logging.info("  %d cities loaded.", len(df))

    logging.info("Running robustness checks ...")
    rob_df = run_robustness(df)

    out_csv = TAB_DIR / "robustness.csv"
    rob_df.to_csv(out_csv, index=False, float_format="%.6f")
    logging.info("Saved: %s", out_csv)

    print("\n=== Robustness results ===")
    print(rob_df[["outcome", "spec", "beta", "ci_lo", "ci_hi", "pval", "r2", "n"]]
          .to_string(index=False, float_format="{:.4f}".format))

    logging.info("Plotting robustness coefficient figure ...")
    plot_robustness(rob_df)

    logging.info("Done.")


if __name__ == "__main__":
    main()
