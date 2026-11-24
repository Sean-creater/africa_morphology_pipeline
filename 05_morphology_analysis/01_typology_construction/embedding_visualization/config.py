"""
config.py
---------
All input/output paths for the embedding visualisation pipeline.

Edit the paths in this file to match your local data layout before running
any of the analysis scripts.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (directory containing this file)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# INPUT PATHS  — edit these to match your data location
# ---------------------------------------------------------------------------

# Continent-scale embedding table (one row per GridID). Not produced by
# any script in this pipeline -- download it from
# ../../../04_morphology_embedding/'s release assets.
# Must contain:
#   embeddings : float32 array of shape (N, 256)
#   grid_ids   : str array of shape (N,)
#   city, country : str arrays of shape (N,)
# Set AFRICA_EMBEDDINGS_PATH to wherever you place the downloaded file.
EMBEDDINGS_PATH = Path(os.environ.get(
    "AFRICA_EMBEDDINGS_PATH",
    "../../../04_morphology_embedding/africa_embeddings.npz",
))

# Parquet table with cluster labels and morphometric features, produced by
# ../cluster_interpretation/01_merge_data.py.
# Required columns: GridID (str), label_k17 (int), plus all feature columns
# listed in FEATURES (see 02_feature_overlay.py).
MODEL_INPUT = (
    ROOT.parent / "cluster_interpretation" / "outputs" / "tables" / "merged_k17.parquet"
)

# ---------------------------------------------------------------------------
# OUTPUT PATHS  — created automatically if they do not exist
# ---------------------------------------------------------------------------

# Intermediate: full-dataset t-SNE coordinates (GridID, tsne1, tsne2, label_k17, …)
TSNE_PARQUET = ROOT / "outputs" / "tsne_coords.parquet"

# Figures
OUT_FIG_DIR = ROOT / "outputs" / "figures"
