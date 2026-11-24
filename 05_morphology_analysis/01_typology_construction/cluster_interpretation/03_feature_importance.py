"""
03_feature_importance.py
------------------------
Rank deduplicated morphometric indicators by their global discriminative
power across the 17 urban-form clusters using η² (eta-squared), computed
after rank-based Inverse Normal Transform (INT).

Method
------
For each selected feature f (from 02_deduplicate_features.py):

1. Apply rank-INT to map the raw, right-skewed distribution to standard
   normal scores.  This places all features on a common scale and makes
   variance decomposition meaningful without assuming normality of the
   original values.

2. Compute η²_f = SS_between / SS_total across the 17 clusters:

       SS_between = Σ_k  n_k · (x̄_k − x̄)²
       SS_total   = Σ_i  (x_i − x̄)²

   η² ∈ [0, 1] is interpreted as the proportion of total variance
   explained by cluster membership.  It is reported as a descriptive
   variance-decomposition ratio; no p-values are computed or reported,
   consistent with the large-N descriptive framework of this study.

Outputs
-------
outputs/tables/feature_importance.csv
    Selected features ranked by η² (descending), with group_id,
    dimension_label, intragroup rank, and η² value.
outputs/figures/feature_importance.pdf
    Horizontal bar chart of all selected features, coloured by
    dimension group.

Usage
-----
    python 03_feature_importance.py [--top N]

    --top   Number of features shown in the bar chart (default: all selected)
"""

import argparse
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.special import ndtri

N_JOBS = 16

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT               = Path(__file__).resolve().parent
MODEL_INPUT        = ROOT / "outputs" / "tables" / "merged_k17.parquet"
SELECTED_CSV       = ROOT / "outputs" / "tables" / "selected_features.csv"
OUT_TABLES         = ROOT / "outputs" / "tables"
OUT_FIGURES        = ROOT / "outputs" / "figures"
CACHE_DIR          = ROOT / "outputs" / "cache"
ETA2_CACHE         = CACHE_DIR / "eta2_selected.parquet"   # {feature, eta2}

# ---------------------------------------------------------------------------
# Rank-based Inverse Normal Transform — vectorised
# ---------------------------------------------------------------------------
def rank_int(series: pd.Series) -> np.ndarray:
    """Blom-offset rank-INT; vectorised, no row-wise Python loop."""
    arr  = pd.to_numeric(series, errors="coerce").values.astype(float)
    mask = np.isfinite(arr)
    n    = mask.sum()
    out  = np.full(len(arr), np.nan)
    if n == 0:
        return out
    sub   = arr[mask]
    order = sub.argsort()
    ranked = np.empty(len(sub))
    ranked[order] = np.arange(1, len(sub) + 1, dtype=float)
    vals_sorted = sub[order]
    i = 0
    while i < len(vals_sorted):
        j = i + 1
        while j < len(vals_sorted) and vals_sorted[j] == vals_sorted[i]:
            j += 1
        if j > i + 1:
            ranked[order[i:j]] = ranked[order[i:j]].mean()
        i = j
    out[mask] = ndtri((ranked - 0.375) / (n + 0.25))
    return out


# ---------------------------------------------------------------------------
# η² computation — vectorised + parallel
# ---------------------------------------------------------------------------
def eta_squared(x: np.ndarray, labels: np.ndarray) -> float:
    """Descriptive η² = SS_between / SS_total; no p-value."""
    mask = np.isfinite(x)
    x, labels = x[mask], labels[mask]
    if len(x) == 0:
        return 0.0
    grand_mean = x.mean()
    ss_total   = ((x - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for k in np.unique(labels):
        xk = x[labels == k]
        ss_between += len(xk) * (xk.mean() - grand_mean) ** 2
    return float(np.clip(ss_between / ss_total, 0.0, 1.0))


def _compute_all_eta2(features: list[str], int_data: dict[str, np.ndarray],
                      labels: np.ndarray) -> dict[str, float]:
    def _one(f):
        return f, eta_squared(int_data[f], labels)
    with ThreadPoolExecutor(max_workers=N_JOBS) as ex:
        results = list(ex.map(_one, features))
    return dict(results)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_key(features: list[str]) -> str:
    """SHA-256 of (model_input mtime, sorted feature list) → hex digest."""
    mtime = MODEL_INPUT.stat().st_mtime
    payload = json.dumps({"mtime": mtime, "features": sorted(features)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_eta2_cache(features: list[str]) -> dict[str, float] | None:
    """Return cached {feature: eta2} if valid, else None."""
    meta = CACHE_DIR / "eta2_selected.meta.json"
    if not ETA2_CACHE.exists() or not meta.exists():
        return None
    stored_key = json.loads(meta.read_text())["key"]
    if stored_key != _cache_key(features):
        return None
    df = pd.read_parquet(ETA2_CACHE)
    return dict(zip(df["feature"], df["eta2"]))


def save_eta2_cache(features: list[str], eta2_vals: dict[str, float]) -> None:
    """Persist eta2 values and a validity key."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": list(eta2_vals), "eta2": list(eta2_vals.values())}).to_parquet(
        ETA2_CACHE, index=False
    )
    meta = CACHE_DIR / "eta2_selected.meta.json"
    meta.write_text(json.dumps({"key": _cache_key(features)}))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_importance(importance: pd.DataFrame, top_n: int, out_path: Path) -> None:
    """
    Horizontal bar chart of selected features ranked by η².

    Bars are coloured by dimension group; a legend maps colours to
    dimension labels.

    Parameters
    ----------
    importance : pd.DataFrame
        Rows = features, sorted by η² descending.
        Required columns: feature, eta2, dimension_label.
    top_n : int
        Number of features to display (top by η²).
    out_path : Path
        Output file path (.pdf).
    """
    subset = importance.head(top_n).iloc[::-1]   # bottom-to-top for barh

    # Assign a colour to each unique dimension
    dims = subset["dimension_label"].unique()
    palette = plt.cm.get_cmap("tab20", len(dims))
    colour_map = {d: palette(i) for i, d in enumerate(dims)}
    colours = [colour_map[d] for d in subset["dimension_label"]]

    fig, ax = plt.subplots(figsize=(7, max(5, len(subset) * 0.28 + 1.5)))
    bars = ax.barh(subset["feature"], subset["eta2"], color=colours, height=0.7)

    # Annotate with η² value
    for bar, val in zip(bars, subset["eta2"]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=7)

    ax.set_xlabel(r"Variance explained by cluster ($\eta^2$)", fontsize=9)
    ax.set_title(
        f"Global discriminative power of morphometric indicators\n"
        f"({top_n} selected features after collinearity deduplication)",
        fontsize=9, pad=8,
    )
    ax.set_xlim(0, subset["eta2"].max() * 1.18)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.tick_params(axis="x", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Legend for dimension groups
    legend_patches = [
        mpatches.Patch(color=colour_map[d], label=d)
        for d in sorted(dims)
    ]
    ax.legend(
        handles=legend_patches,
        title="Dimension group",
        fontsize=7,
        title_fontsize=7.5,
        loc="lower right",
        framealpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Figure saved: %s (+ .png)", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(top_n: int | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    # --- Load selected features ---
    selected = pd.read_csv(SELECTED_CSV)
    features = selected["feature"].tolist()
    logging.info("Selected features to rank: %d", len(features))

    # --- η²: load from cache or recompute ---
    eta2_vals = load_eta2_cache(features)
    if eta2_vals is not None:
        logging.info("η² loaded from cache (%s).", ETA2_CACHE)
    else:
        logging.info("Cache miss — loading model input ...")
        df = pd.read_parquet(MODEL_INPUT, columns=features + ["label_k17"])
        labels = df["label_k17"].values

        logging.info("Applying rank-INT to %d features (parallel, %d workers) ...",
                     len(features), N_JOBS)
        def _rint(f):
            return f, rank_int(df[f])
        with ThreadPoolExecutor(max_workers=N_JOBS) as ex:
            int_data = dict(ex.map(_rint, features))

        logging.info("Computing η² ...")
        eta2_vals = _compute_all_eta2(features, int_data, labels)
        save_eta2_cache(features, eta2_vals)
        logging.info("η² cached to %s.", ETA2_CACHE)

    # --- Merge with group metadata and sort ---
    importance = selected[["feature", "group_id", "dimension_label",
                            "intragroup_rank"]].copy()
    importance["eta2"] = importance["feature"].map(eta2_vals)
    importance = importance.sort_values("eta2", ascending=False).reset_index(drop=True)
    importance.insert(0, "global_rank", range(1, len(importance) + 1))

    out_csv = OUT_TABLES / "feature_importance.csv"
    importance.to_csv(out_csv, index=False)
    logging.info("Importance table saved: %s", out_csv)

    logging.info("Top 10 features by η²:")
    for _, row in importance.head(10).iterrows():
        logging.info(
            "  %2d. %-20s  η²=%.4f  [%s]",
            row["global_rank"], row["feature"], row["eta2"], row["dimension_label"],
        )

    # --- Plot ---
    n_plot = top_n if top_n is not None else len(importance)
    plot_importance(importance, top_n=n_plot,
                    out_path=OUT_FIGURES / "feature_importance.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rank selected features by η² after rank-INT normalisation."
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Number of features to plot (default: all selected features).",
    )
    args = parser.parse_args()
    main(top_n=args.top)
