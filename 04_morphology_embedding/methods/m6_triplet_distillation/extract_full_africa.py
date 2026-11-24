"""
M6: Extract embeddings for the full African grid-cell dataset.

Reads every *_v15.csv under CLEANED_CSV_ROOT, runs the best M6 fold model,
and saves per-city .npz files mirroring the country/city directory structure.
This applies the trained M6 student to the full production dataset, well
beyond the ~5,494-image analysis_set used for the methodological comparison
itself — only tabular morphology features are required, no imagery.

Output layout:
  OUTPUT_ROOT/{Country}/{City}.npz
    embeddings : (N, 256) float32, L2-normalised
    grid_ids   : (N,)     str

Skips cities whose output file already exists (safe to resume).

OUTPUT_ROOT defaults to a directory next to CLEANED_CSV_ROOT; override with
the MORPHOLOGY_EMBEDDINGS_ROOT environment variable.
"""
import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.paths import CLEANED_CSV_ROOT
sys.path.insert(0, str(Path(__file__).parent))
from models import StudentMLP

OUTPUT_ROOT = Path(os.environ.get(
    "MORPHOLOGY_EMBEDDINGS_ROOT",
    str(CLEANED_CSV_ROOT.parent / "data_embeddings_full_africa"),
))
MODEL_DIR   = Path(__file__).parent.parent.parent / "outputs/m6_triplet_distillation/models"
CSV_SUFFIX  = "_grid_250m_hex_enriched_metrics_v15.csv"
BATCH_SIZE  = 4096

# Columns where NaN means the feature is genuinely absent (fill with 0).
# Land cover ratios, building height/volume, and all street-network topology metrics.
_ZERO_FILL_COLS = frozenset([
    # Land cover (LC_ / LCF_)
    "LC_Shrub","LC_Herb","LC_Crop","LC_Built","LC_Bare","LC_Water","LC_Wetlnd",
    "LC_Snow","LC_Moss","LC_Ocean","LC_Shannon",
    "LC_FCDcBr","LC_FCDcNd","LC_FCEvBr","LC_FCEvNd","LC_FCMix","LC_FCUnd",
    "LC_FODcBr","LC_FODcNd","LC_FOEvBr","LC_FOEvNd","LC_FOMix","LC_FOUnd",
    "LCF_Bare","LCF_Built","LCF_Crop","LCF_Grass","LCF_Moss",
    "LCF_SeaWtr","LCF_Shrub","LCF_Snow","LCF_Tree","LCF_Water",
    # Building height / volume (absent when no buildings)
    "H_max","H_mean","H_std","H_sum",
    "V_max","V_mean","V_std","V_sum",
    # Street network topology (absent when NodeDens~0, no network formed)
    "BtwCentM","BtwCentS","BtwCentX",
    "ClsCentM","ClsCentS","ClsCentX",
    "ClusterM","ClusterS",
    "COINSCnt","COINSLen",
    "CuldesacL","CuldesacR","Cyclomat",
    "DeadendCnt","DeadendDen",
    "EdgeBtwM","EdgeClsM","EdgeNdRat","EdgeStrM",
    "FourWayCnt","ThreeWayCnt","InterCnt","InterDens",
    "MeanDeg","MeanNdDst","Meshedness",
    "NdNbDenM","NdNbDenS","StdDeg",
    "GammaIdx","Ratio3way","Ratio4way",
    "StrCentM","StrCentS","StrCentX",
    "StrLinM","StrLinS","StrOpen","StrOrientM","StrOrientS",
    "StrWidth","StrWidthD","StreetsPN","StreetDens","SegmentDen",
])


def setup_logging():
    log_path = OUTPUT_ROOT / "extract.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger()


def load_best_model(model_dir):
    best = None
    for p in sorted(Path(model_dir).glob("student_fold*.pth")):
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if best is None or ckpt["best_val_loss"] < best[1]:
            best = (p, ckpt["best_val_loss"], ckpt)
    log = logging.getLogger()
    log.info(f"Best fold: {best[0].name}  (val_loss={best[1]:.4f})")
    ckpt = best[2]
    model = StudentMLP(ckpt["input_dim"], ckpt["hidden_dims"],
                       ckpt["dropout"], ckpt["batch_norm"], ckpt["output_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["scaler"], ckpt["feature_names"]


_MEDIAN_FILL_COLS = None  # populated at runtime from feature_names - _ZERO_FILL_COLS


def fill_nan(df, feature_names):
    for col in feature_names:
        if not df[col].isna().any():
            continue
        if col in _ZERO_FILL_COLS:
            df[col] = df[col].fillna(0.0)
        # median-fill cols: handled by dropping rows below
    return df


def embed_csv(csv_path, model, scaler, feature_names, median_fill_cols, device):
    df = pd.read_csv(csv_path, low_memory=False)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Apply zero-fill for absence-coded columns
    df = fill_nan(df, feature_names)

    # Drop rows where any building/plot morphology metric is still NaN
    n_before = len(df)
    df = df.dropna(subset=median_fill_cols)
    n_dropped = n_before - len(df)

    if len(df) == 0:
        return None, None, n_dropped

    X = scaler.transform(df[feature_names].values.astype(np.float32)).clip(-10, 10)
    X = X.astype(np.float32)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            batch = torch.from_numpy(X[i:i + BATCH_SIZE]).to(device)
            embeddings.append(model(batch).cpu().numpy())

    return np.vstack(embeddings), df["GridID"].astype(str).values, n_dropped


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log = setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model, scaler, feature_names = load_best_model(MODEL_DIR)
    model = model.to(device)
    median_fill_cols = [c for c in feature_names if c not in _ZERO_FILL_COLS]
    log.info(f"Model: {len(feature_names)} features -> 256-dim  "
             f"(zero-fill: {len(_ZERO_FILL_COLS)}, drop-if-NaN: {len(median_fill_cols)})")

    csv_paths = sorted(Path(CLEANED_CSV_ROOT).rglob(f"*{CSV_SUFFIX}"))
    log.info(f"Found {len(csv_paths):,} CSVs under {CLEANED_CSV_ROOT}")

    t0 = time.time()
    n_done = n_skip = n_err = 0
    total_cells = total_dropped = 0

    for csv_path in csv_paths:
        country = csv_path.parent.name
        city    = csv_path.name.replace(CSV_SUFFIX, "")
        out_dir = OUTPUT_ROOT / country
        out_path = out_dir / f"{city}.npz"

        if out_path.exists():
            n_skip += 1
            continue

        try:
            embeddings, grid_ids, n_dropped = embed_csv(
                csv_path, model, scaler, feature_names, median_fill_cols, device)
            total_dropped += n_dropped

            if embeddings is None:
                log.warning(f"  ALL ROWS DROPPED {country}/{city} ({n_dropped} rows)")
                n_err += 1
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, embeddings=embeddings, grid_ids=grid_ids)
            total_cells += len(grid_ids)
            n_done += 1

            if n_done % 500 == 0:
                elapsed = time.time() - t0
                rate = total_cells / elapsed
                remaining = len(csv_paths) - n_done - n_skip - n_err
                log.info(f"  [{n_done}/{len(csv_paths)}] {total_cells:,} cells  "
                         f"dropped={total_dropped:,}  "
                         f"{rate:.0f} cells/s  ~{remaining * (elapsed/n_done)/60:.0f} min left")

        except Exception as e:
            log.warning(f"  SKIP {country}/{city}: {e}")
            n_err += 1

    elapsed = time.time() - t0
    log.info(f"\nDone in {elapsed/60:.1f} min")
    log.info(f"  Processed: {n_done}  Skipped (existing): {n_skip}  Errors/empty: {n_err}")
    log.info(f"  Total cells embedded: {total_cells:,}")
    log.info(f"  Total cells dropped (NaN):  {total_dropped:,}")
    log.info(f"  Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
