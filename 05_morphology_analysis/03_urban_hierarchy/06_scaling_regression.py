"""
06_scaling_regression.py
========================
Continent-wide standardized OLS scaling regressions for H17*, r17, and
r17_z (permutation-null z-score).

Three models per outcome:
  M1 (raw)      : Y_z ~ X_z                          (OLS, country-clustered SE)
  M2 (adjusted) : Y_z ~ X_z + community_FE           (OLS, country-clustered SE)
  M3 (size-ctrl): Y_z ~ X_z + log(size)              (OLS, country-clustered SE)
                  size = n_cells for H17_star, n_edges for r17/r17_z —
                  H17_star is mechanically bounded by n_cells (a city with
                  n_cells < 17 cannot show all 17 types, so its ceiling is
                  log(n_cells)/log(17)), and log(population) correlates
                  r=0.77 with log(n_cells) here, so part of the raw
                  population-H17_star slope could mechanically reflect
                  bigger cities simply having more grid cells. M3 checks
                  whether the slope survives controlling for that.

                  For r17, n_edges is NOT a clean ceiling the way n_cells
                  is for H17_star — it's a composite of cell count, graph
                  topology AND population-driven spatial extent, so
                  "controlling" for it is a conditional association, not
                  confound removal (the M3 sign flip for raw r17, below,
                  should NOT be read as proof the raw relationship is an
                  artefact). r17_z is the more direct test: a
                  permutation-null z-score computed in
                  01_compute_city_metrics.py by holding each city's own
                  graph structure AND 17-type count distribution fixed,
                  permuting which node gets which type label, and scoring
                  the observed r17 against that null — this already nets
                  out graph size/topology/composition at the single-city
                  level, so regressing r17_z against population directly
                  answers "given this city's actual graph, is its spatial
                  clustering stronger/weaker than chance" without needing
                  a second-stage covariate adjustment. M3 for r17_z
                  (+log(n_edges) again) is then a check for RESIDUAL
                  size-dependence after the null adjustment, not the
                  primary test.

Where:
  Y_z = (Y - mean(Y)) / sd(Y)     [outcome, standardized]
  X_z = (log_pop - mean) / sd     [standardized log population]

Output
------
  outputs/tables/scaling_regression.csv   — coefficient table
  outputs/figures/scaling_coef.pdf/png    — coefficient plot (main text figure)

Usage
-----
    python 06_scaling_regression.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


matplotlib.rcParams.update({
    "font.family":     "Arial",
    "font.size":       8,
    "svg.fonttype": "none",
})

ROOT      = Path(__file__).resolve().parent
METRICS   = ROOT / "outputs" / "tables" / "city_metrics.parquet"
FIG_DIR   = ROOT / "outputs" / "figures"
TAB_DIR   = ROOT / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4

# Community geographic names (original ID 0–7 → name)
COMM_NAMES = {
    0: "West African Inland",
    1: "Gulf of Guinea Coastal",
    2: "Southern–East African",
    3: "East–Central African",
    4: "Sahel–Central African Interior",
    5: "North African–Nile",
    6: "Nile Urban Corridor",
    7: "East African Highlands",
}

OUTCOMES = {
    "H17_star": "Morphological diversity H* (k=17)",
    "r17":      "Spatial assortativity r (k=17)",
    "r17_z":    "Assortativity, permutation-null z (k=17)",
}

# Mechanical "sample size" each outcome is bounded by — see module
# docstring (M3). H17_star's ceiling is set by how many grid cells a city
# has (can't show more types than cells); r17/r17_z are computed from the
# observed Queen-adjacency edge pairs, so n_edges plays the analogous role
# (though, per the module docstring, n_edges is a much less clean size
# proxy for r17 than n_cells is for H17_star).
SIZE_COVARIATE = {
    "H17_star": "n_cells",
    "r17":      "n_edges",
    "r17_z":    "n_edges",
}

# Colours for the outcomes in coefficient plot
OUTCOME_COLORS = {
    "H17_star": "#2166AC",  # blue
    "r17":      "#D6604D",  # red-orange
    "r17_z":    "#8B4A9C",  # purple — visually distinct from raw r17
}


# ---------------------------------------------------------------------------
# OLS with country-clustered SE
# ---------------------------------------------------------------------------
def run_ols(df, formula, cluster_var="country"):
    """Fit OLS and return country-clustered SE results."""
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster_var]},
    )
    return model


def extract_coef(result, param="X_z"):
    """Extract β, 95% CI, R², p-value for a named parameter."""
    beta   = result.params[param]
    ci_lo  = result.conf_int().loc[param, 0]
    ci_hi  = result.conf_int().loc[param, 1]
    pval   = result.pvalues[param]
    r2     = result.rsquared
    r2_adj = result.rsquared_adj
    n      = int(result.nobs)
    return dict(beta=beta, ci_lo=ci_lo, ci_hi=ci_hi,
                pval=pval, r2=r2, r2_adj=r2_adj, n=n)


# ---------------------------------------------------------------------------
# Main regression loop
# ---------------------------------------------------------------------------
def run_regressions(df):
    records = []

    for outcome, outcome_label in OUTCOMES.items():
        df_sub = df.dropna(subset=[outcome, "log_pop", "community", "country"]).copy()

        # Standardize X and Y
        mu_x, sd_x = df_sub["log_pop"].mean(), df_sub["log_pop"].std()
        mu_y, sd_y = df_sub[outcome].mean(),   df_sub[outcome].std()
        df_sub["X_z"] = (df_sub["log_pop"] - mu_x) / sd_x
        df_sub["Y_z"] = (df_sub[outcome]   - mu_y) / sd_y

        n_countries = df_sub["country"].nunique()
        logging.info("  %s: n=%d cities, %d countries, SD(Y)=%.4f",
                     outcome, len(df_sub), n_countries, sd_y)

        # M1: raw (no community FE)
        m1 = run_ols(df_sub, "Y_z ~ X_z")
        c1 = extract_coef(m1, "X_z")
        c1.update(outcome=outcome, model="M1_raw",
                  sd_x=sd_x, sd_y=sd_y, n_countries=n_countries)
        records.append(c1)
        logging.info("    M1 raw:      β=%.4f [%.4f, %.4f]  R²=%.3f  p=%.2e",
                     c1["beta"], c1["ci_lo"], c1["ci_hi"], c1["r2"], c1["pval"])

        # M2: community fixed effects
        m2 = run_ols(df_sub, "Y_z ~ X_z + C(community)")
        c2 = extract_coef(m2, "X_z")
        c2.update(outcome=outcome, model="M2_community_FE",
                  sd_x=sd_x, sd_y=sd_y, n_countries=n_countries)
        records.append(c2)
        logging.info("    M2 comm FE:  β=%.4f [%.4f, %.4f]  R²=%.3f  p=%.2e",
                     c2["beta"], c2["ci_lo"], c2["ci_hi"], c2["r2"], c2["pval"])

        # M3: control for the outcome's own mechanical "sample size"
        # (n_cells for H17_star, n_edges for r17) — see module docstring.
        size_col = SIZE_COVARIATE[outcome]
        df_sub3 = df_sub[df_sub[size_col] > 0].copy()
        df_sub3["log_size"] = np.log(df_sub3[size_col])
        m3 = run_ols(df_sub3, "Y_z ~ X_z + log_size")
        c3 = extract_coef(m3, "X_z")
        c3.update(outcome=outcome, model=f"M3_log_{size_col}",
                  sd_x=sd_x, sd_y=sd_y, n_countries=df_sub3["country"].nunique())
        records.append(c3)
        logging.info("    M3 +log(%s): β=%.4f [%.4f, %.4f]  R²=%.3f  p=%.2e",
                     size_col, c3["beta"], c3["ci_lo"], c3["ci_hi"], c3["r2"], c3["pval"])

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Coefficient plot
# ---------------------------------------------------------------------------
def plot_coef(coef_df):
    """
    One panel per outcome (dynamic — currently H17_star, r17, r17_z), each
    showing M1/M2/M3. Points = β, horizontal bars = 95% CI.
    """
    n_out = len(OUTCOMES)
    fig, axes = plt.subplots(1, n_out, figsize=(70 * n_out * MM, 70 * MM),
                             sharey=False)
    if n_out == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    model_labels = {"M1_raw": "Raw OLS", "M2_community_FE": "Community FE",
                    "M3_log_n_cells": "+ log(n_cells)",
                    "M3_log_n_edges": "+ log(n_edges)"}
    model_markers = {"M1_raw": "o", "M2_community_FE": "s",
                     "M3_log_n_cells": "^", "M3_log_n_edges": "^"}
    model_offsets = {"M1_raw": 0.22, "M2_community_FE": 0.0,
                     "M3_log_n_cells": -0.22, "M3_log_n_edges": -0.22}  # y jitter

    for ax, (outcome, outcome_label) in zip(axes, OUTCOMES.items()):
        col = OUTCOME_COLORS[outcome]
        sub = coef_df[coef_df["outcome"] == outcome].copy()

        for i, row in enumerate(sub.itertuples()):
            yo = model_offsets[row.model]
            ax.errorbar(
                row.beta, i + yo,
                xerr=[[row.beta - row.ci_lo], [row.ci_hi - row.beta]],
                fmt=model_markers[row.model],
                color=col, ecolor=col, elinewidth=1.2,
                markersize=5, capsize=3, zorder=4,
                label=model_labels[row.model] if i == 0 else "",
            )
            # Annotate β and R²
            ax.text(row.ci_hi + 0.005, i + yo,
                    f"β={row.beta:.3f}, R²={row.r2:.3f}",
                    va="center", ha="left", fontsize=5.5, color="#444444")

        ax.axvline(0, color="#AAAAAA", lw=0.8, ls="--", zorder=1)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(
            [model_labels[m] for m in sub["model"]], fontsize=6.5)
        ax.set_xlabel("Standardised β (95% CI)", fontsize=7)
        ax.set_title(outcome_label, fontsize=7.5, fontweight="bold", pad=5)

        # Style
        ax.set_facecolor("white")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color("#AAAAAA")
            ax.spines[sp].set_linewidth(0.6)
        ax.tick_params(colors="#555555", labelsize=6.5)
        ax.xaxis.label.set_color("#555555")

    fig.tight_layout(pad=1.5, w_pad=3.0)
    for ext in ("svg", "png"):
        fig.savefig(FIG_DIR / f"scaling_coef.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Saved: scaling_coef.pdf/png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logging.info("Loading city metrics ...")
    df = pd.read_parquet(METRICS)
    df = df.dropna(subset=["log_pop", "community"]).copy()
    df["community"] = df["community"].astype(int)
    logging.info("  %d cities loaded.", len(df))

    logging.info("Running continent-wide regressions ...")
    coef_df = run_regressions(df)

    # Save table
    out_csv = TAB_DIR / "scaling_regression.csv"
    coef_df.to_csv(out_csv, index=False, float_format="%.6f")
    logging.info("Saved: %s", out_csv)

    # Pretty print
    print("\n=== Scaling regression results ===")
    display_cols = ["outcome", "model", "beta", "ci_lo", "ci_hi",
                    "pval", "r2", "r2_adj", "n", "n_countries"]
    print(coef_df[display_cols].to_string(index=False, float_format="{:.4f}".format))

    logging.info("Plotting coefficient figure ...")
    plot_coef(coef_df)

    logging.info("Done.")


if __name__ == "__main__":
    main()
