"""
01_tsne.py
----------
Full-dataset t-SNE on all ~1 M M6 StudentMLP embeddings, followed by a
publication-quality cluster scatter plot.

Pipeline
--------
1. Load all per-city .npz embedding files (embeddings, grid_ids).
2. Pre-clean: remove cells whose L2 norm falls outside
   [Q1 − 5·IQR, Q3 + 5·IQR] to filter degenerate embeddings.
3. L2-normalise (cosine-equivalent, consistent with triplet training).
4. Full t-SNE via openTSNE (Barnes-Hut, perplexity=50, cosine metric).
   Runtime: ~30 min on a 32-core CPU.
5. Join k=17 cluster labels from merged_k17.parquet.
6. Post-clean: drop cells whose t-SNE coordinate deviates > 5σ on
   either axis.
7. Save full coordinates to outputs/tsne_coords.parquet (cached —
   re-running the script skips steps 1–7 if the file exists).
8. Plot all ~1 M cells; annotate each cluster with its index.

Output
------
outputs/tsne_coords.parquet
outputs/figures/tsne_k17.pdf
outputs/figures/tsne_k17.png

Usage
-----
    python 01_tsne.py

Paths
-----
All input/output paths are defined in config.py.
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import normalize
from openTSNE import TSNE

from config import EMBEDDINGS_PATH, MODEL_INPUT, TSNE_PARQUET, OUT_FIG_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PRE_CLEAN_IQR_FACTOR = 5.0
POST_CLEAN_SIGMA     = 5.0

TSNE_PARAMS = dict(
    n_components=2,
    perplexity=100,           # higher perplexity → tighter, more separated clusters
    early_exaggeration=24,    # stronger early push → more compact clusters
    metric="cosine",
    n_jobs=-1,                # use all CPU cores
    random_state=42,
    verbose=True,
)

# ---------------------------------------------------------------------------
# Colour palette — 17 qualitative colours (cluster 1 → index 0, …)
# ---------------------------------------------------------------------------
PALETTE_17 = [
    "#DBDBDB", "#FEEFCF", "#FCBBC5", "#F5E6C8", "#E5152C",
    "#FFFF4C", "#5BA04A", "#FBB732", "#6F391F", "#B0E1E7",
    "#7B67EE", "#E89679", "#58C4EC", "#828F9B", "#1C1C5F",
    "#992FA3", "#C49A3C",
]

FIG_W_MM, FIG_H_MM = 150, 150
MM_TO_IN = 1 / 25.4


# ---------------------------------------------------------------------------
# Step 1 — Load embeddings
# ---------------------------------------------------------------------------
def load_all_embeddings():
    t0 = time.time()
    log.info("Loading embeddings from %s ...", EMBEDDINGS_PATH)
    f = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    embeddings = f["embeddings"].astype(np.float32)
    grid_ids   = f["grid_ids"].astype(str)
    cities     = f["city"].astype(str)
    countries  = f["country"].astype(str)
    n_cities = len({(c, ci) for c, ci in zip(countries, cities)})
    log.info("Loaded %s cells from %d cities  [%.1fs]",
             f"{len(embeddings):,}", n_cities, time.time() - t0)
    return embeddings, grid_ids, cities, countries


# ---------------------------------------------------------------------------
# Step 2–3 — Pre-clean + L2 normalise
# ---------------------------------------------------------------------------
def pre_clean(embeddings, grid_ids, cities, countries):
    norms = np.linalg.norm(embeddings, axis=1)
    q1, q3 = np.percentile(norms, [25, 75])
    iqr = q3 - q1
    lo  = q1 - PRE_CLEAN_IQR_FACTOR * iqr
    hi  = q3 + PRE_CLEAN_IQR_FACTOR * iqr
    mask = (norms >= lo) & (norms <= hi) & np.isfinite(norms)
    log.info("Pre-clean: removed %d cells (L2 norm outside [%.3f, %.3f])",
             (~mask).sum(), lo, hi)
    return embeddings[mask], grid_ids[mask], cities[mask], countries[mask]


# ---------------------------------------------------------------------------
# Step 4 — Full t-SNE
# ---------------------------------------------------------------------------
def run_tsne(embeddings):
    log.info("Running t-SNE on %s cells (perplexity=%d, metric=%s) ...",
             f"{len(embeddings):,}", TSNE_PARAMS["perplexity"], TSNE_PARAMS["metric"])
    t0     = time.time()
    tsne   = TSNE(**TSNE_PARAMS)
    coords = tsne.fit(embeddings)
    log.info("t-SNE done: shape %s  [%.1fs]", coords.shape, time.time() - t0)
    return np.array(coords, dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 5–6 — Join labels + post-clean
# ---------------------------------------------------------------------------
def join_and_clean(grid_ids, cities, countries, coords):
    labels_df = pd.read_parquet(MODEL_INPUT, columns=["GridID", "label_k17"])
    labels_df["GridID"] = labels_df["GridID"].astype(str)
    df = pd.DataFrame({
        "GridID":  grid_ids,
        "city":    cities,
        "country": countries,
        "tsne1":   coords[:, 0],
        "tsne2":   coords[:, 1],
    })
    df = df.merge(labels_df, on="GridID", how="left")
    missing = df["label_k17"].isna().sum()
    if missing:
        log.warning("%d cells missing label — dropping", missing)
        df = df.dropna(subset=["label_k17"])
    df["label_k17"] = df["label_k17"].astype(int)

    for axis in ("tsne1", "tsne2"):
        mu, s = df[axis].mean(), df[axis].std()
        mask  = (df[axis] - mu).abs() <= POST_CLEAN_SIGMA * s
        removed = (~mask).sum()
        if removed:
            log.info("Post-clean %s: removed %d cells beyond %.1fσ",
                     axis, removed, POST_CLEAN_SIGMA)
        df = df[mask]

    log.info("After join + post-clean: %s cells", f"{len(df):,}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 7 — Save parquet
# ---------------------------------------------------------------------------
def save_parquet(df):
    TSNE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    log.info("Saving %s ...", TSNE_PARQUET)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        TSNE_PARQUET, compression="snappy",
    )
    log.info("Saved %s rows", f"{len(df):,}")


# ---------------------------------------------------------------------------
# Step 8 — Plot
# ---------------------------------------------------------------------------
def plot(df):
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size":   7,
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
    })

    clusters  = sorted(df["label_k17"].unique())
    color_map = {c: PALETTE_17[i % len(PALETTE_17)] for i, c in enumerate(clusters)}

    fig, ax = plt.subplots(
        figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN), dpi=600)
    fig.patch.set_facecolor("white")

    # Draw largest clusters first so smaller ones appear on top
    draw_order = df["label_k17"].value_counts().sort_values(
        ascending=False).index.tolist()
    for c in draw_order:
        sub = df[df["label_k17"] == c]
        ax.scatter(sub["tsne1"], sub["tsne2"],
                   c=color_map[c], s=0.35, alpha=0.35,
                   linewidths=0, rasterized=True, zorder=2)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Cluster index labels: hollow circle + number at cluster median
    for c in clusters:
        sub = df[df["label_k17"] == c]
        cx, cy = sub["tsne1"].median(), sub["tsne2"].median()
        ax.scatter(cx, cy, s=80, color="none", zorder=5,
                   linewidths=0.5, edgecolors="black")
        ax.text(cx, cy, str(c), fontsize=5, fontweight="bold",
                color="black", ha="center", va="center", zorder=6,
                fontfamily="Arial")

    plt.tight_layout()

    path = OUT_FIG_DIR / "tsne_k17.png"
    fig.savefig(path, dpi=600, bbox_inches="tight", format="png")
    log.info("Saved: %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=== 01_tsne.py ===")

    if TSNE_PARQUET.exists():
        log.info("Cache found — loading %s", TSNE_PARQUET)
        df = pq.read_table(TSNE_PARQUET).to_pandas()
    else:
        embeddings, grid_ids, cities, countries = load_all_embeddings()
        embeddings, grid_ids, cities, countries = pre_clean(
            embeddings, grid_ids, cities, countries)
        embeddings = normalize(embeddings, norm="l2").astype(np.float32)
        coords = run_tsne(embeddings)
        df     = join_and_clean(grid_ids, cities, countries, coords)
        save_parquet(df)

    log.info("Plotting full dataset: %s pts", f"{len(df):,}")
    plot(df)
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
