"""
02_deduplicate_features.py
--------------------------
Identify redundant morphometric indicators via Spearman correlation with
complete-linkage hierarchical clustering, and retain the single top
representative per collinear group.

Method
------
1. Compute the full 164×164 Spearman rank-correlation matrix.
2. Convert to a distance matrix: d_ij = 1 − |ρ_ij|.
3. Apply complete-linkage hierarchical clustering.  Cut the tree at
   distance threshold t = 1 − THRESHOLD (default THRESHOLD=0.90, so t=0.10).
   Complete linkage guarantees that *every pair* within a group satisfies
   |ρ| ≥ THRESHOLD — avoiding the chaining artefact of connected-component
   (single-linkage) methods.
4. Within each group, rank members by η² (descriptive ANOVA variance ratio,
   computed on rank-INT values) and retain only the top-1 representative.
   Singleton components are retained as-is.
5. Each group is assigned a dimension label naming its top-ranked (η²)
   member, e.g. "Group 3 (BldgDens)"; singleton groups are labelled
   "Singleton".

Outputs
-------
outputs/tables/feature_groups.csv
    All 164 features with group ID, intra-group η² rank, selection flag,
    η² value, and dimension label.
outputs/tables/selected_features.csv
    Deduplicated feature list (one representative per group; selected == True).
outputs/tables/dimension_labels.json
    {feature: dimension_label} mapping for all 164 features.

Usage
-----
    python 02_deduplicate_features.py [--threshold F]

    --threshold  Spearman |ρ| cutoff for grouping (default: 0.90).
                 Supplementary sensitivity check: also run at 0.85 and 0.95.
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.special import ndtri

N_JOBS = 16

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent
MODEL_INPUT  = ROOT / "outputs" / "tables" / "merged_k17.parquet"
OUT_TABLES   = ROOT / "outputs" / "tables"

# ---------------------------------------------------------------------------
# Feature columns (164 morphometric indicators, M6 input space)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "LC_Shrub","LC_Herb","LC_Crop","LC_Built","LC_Bare","LC_Snow","LC_Water",
    "LC_Wetlnd","LC_Moss","LC_FCEvNd","LC_FCEvBr","LC_FCDcNd","LC_FCDcBr",
    "LC_FCMix","LC_FCUnd","LC_FOEvNd","LC_FOEvBr","LC_FODcNd","LC_FODcBr",
    "LC_FOMix","LC_FOUnd","LC_Ocean","H_max","H_mean","H_std","H_sum",
    "V_sum","V_mean","V_max","V_std","LCF_Bare","LCF_Built",
    "LCF_Crop","LCF_Grass","LCF_Moss","LCF_Water","LCF_SeaWtr","LCF_Shrub",
    "LCF_Snow","LCF_Tree","LC_Shannon","Mean_Area","Max_Area","Std_Area",
    "Mean_Perim","Max_Perim","Std_Perim","Mean_Court","Std_Court","Mean_Longe",
    "Max_Longe","Std_Longe","Mean_Circu","Std_Circu","Mean_Squ_1","Std_Squ_1",
    "Mean_Compa","Std_Compa","Mean_Equiv","Std_Equiv","Mean_Recta","Std_Recta",
    "Mean_Conve","Std_Conve","Mean_Elong","Std_Elong","Mean_Squar","Std_Squar",
    "Mean_Centr","Std_Centr","Mean_CentS","Std_CentS","Mean_CrtYd","Std_CrtYd",
    "Mean_Facad","Std_Facad","Mean_Fract","Std_Fract","Mean_ShpId","Std_ShpId",
    "Mean_FormF","Std_FormF","Mean_H2F","Std_H2F","Mean_VolDe","Std_VolDe",
    "Mean_Corne","Max_Corne","Std_Corne","Mean_Orien","Std_Orien","Mean_Share",
    "Max_Share","Std_Share","BldgCount","BldgDens","CoverRatio","OpenRatio",
    "PerimDens","CourtDens","ShareDens","FloorARat","Mean_Align","Std_Align",
    "Mean_IBDis","Std_IBDis","Mean_NbDis","Std_NbDis","Mean_Neigh","Max_Neigh",
    "Std_Neigh","Mean_BAdj","Std_BAdj","StreetDens","NodeDens","StreetsPN",
    "SegmentDen","PlotDens","BldgDensN","PlotFrag","StrWidth","StrOpen",
    "StrWidthD","StrLinM","StrLinS","StrOrientM","StrOrientS","BldgStrAM",
    "BldgStrAS","BtwCentM","BtwCentX","BtwCentS","ClsCentM","ClsCentX",
    "ClsCentS","StrCentM","StrCentX","StrCentS","NdNbDenM","NdNbDenS",
    "ClusterM","ClusterS","MeanDeg","StdDeg","DeadendCnt","InterCnt",
    "ThreeWayCnt","FourWayCnt","Ratio3way","Ratio4way","CuldesacR","MeanNdDst",
    "Cyclomat","EdgeNdRat","GammaIdx","Meshedness","EdgeBtwM","EdgeClsM",
    "EdgeStrM","CuldesacL","COINSCnt","COINSLen","DeadendDen","InterDens",
]


# ---------------------------------------------------------------------------
# Rank-based Inverse Normal Transform (Blom offset) — vectorised
# ---------------------------------------------------------------------------
def rank_int(series: pd.Series) -> np.ndarray:
    """
    z_i = Φ⁻¹( (rank_i − 3/8) / (n + 1/4) )

    Vectorised: no row-wise Python loop.  NaN values are preserved.
    """
    arr    = pd.to_numeric(series, errors="coerce").values.astype(float)
    mask   = np.isfinite(arr)
    n      = mask.sum()
    out    = np.full(len(arr), np.nan)
    if n == 0:
        return out
    sub    = arr[mask]
    # average rank (ties get mean rank)
    order  = sub.argsort()
    ranked = np.empty(len(sub))
    ranked[order] = np.arange(1, len(sub) + 1, dtype=float)
    # handle ties: find runs of equal values
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


def rank_int_df(df: pd.DataFrame, features: list[str], n_jobs: int = N_JOBS) -> dict[str, np.ndarray]:
    """Apply rank_int to all features in parallel using ThreadPoolExecutor."""
    def _one(f):
        return f, rank_int(df[f])
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        results = list(ex.map(_one, features))
    return dict(results)


# ---------------------------------------------------------------------------
# η² (descriptive variance ratio) — vectorised
# ---------------------------------------------------------------------------
def eta_squared(x: np.ndarray, labels: np.ndarray) -> float:
    """
    η² = SS_between / SS_total.

    Computed on rank-INT values as a descriptive ratio only; no p-value
    is derived.  Spatial autocorrelation affects hypothesis testing but
    not this descriptive variance decomposition.
    """
    mask           = np.isfinite(x)
    x, labels      = x[mask], labels[mask]
    if len(x) == 0:
        return 0.0
    grand_mean     = x.mean()
    ss_total       = ((x - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    unique_k = np.unique(labels)
    ss_between = 0.0
    for k in unique_k:
        xk = x[labels == k]
        ss_between += len(xk) * (xk.mean() - grand_mean) ** 2
    return float(np.clip(ss_between / ss_total, 0.0, 1.0))


def eta_squared_all(int_vals: dict[str, np.ndarray], labels_arr: np.ndarray,
                    features: list[str], n_jobs: int = N_JOBS) -> dict[str, float]:
    """Compute η² for all features in parallel."""
    def _one(f):
        return f, eta_squared(int_vals[f], labels_arr)
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        results = list(ex.map(_one, features))
    return dict(results)


# ---------------------------------------------------------------------------
# Complete-linkage collinearity grouping
# ---------------------------------------------------------------------------
def build_collinearity_groups(
    corr_abs: pd.DataFrame,
    features: list[str],
    threshold: float,
) -> list[list[str]]:
    """
    Group features by complete-linkage hierarchical clustering of the
    Spearman dissimilarity matrix d_ij = 1 − |ρ_ij|.

    Complete linkage ensures that every pair within a group has |ρ| ≥ threshold,
    eliminating the chaining artefact of connected-component / single-linkage
    methods.

    Returns
    -------
    list of lists
        Each inner list is one collinear group (feature names).  Members are
        NOT sorted here — caller sorts by η².
    """
    dist_condensed = []
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            dist_condensed.append(1.0 - corr_abs.iloc[i, j])

    dist_condensed = np.array(dist_condensed, dtype=float)
    # Clip numerical noise; replace NaN (zero-variance / undefined corr) with max distance 1.0
    dist_condensed = np.where(np.isfinite(dist_condensed), dist_condensed, 1.0)
    dist_condensed = np.clip(dist_condensed, 0.0, 1.0)

    Z = linkage(dist_condensed, method="complete")
    cut_distance = 1.0 - threshold          # e.g. 0.10 for threshold=0.90
    cluster_ids  = fcluster(Z, t=cut_distance, criterion="distance")

    groups: dict[int, list[str]] = {}
    for feat, cid in zip(features, cluster_ids):
        groups.setdefault(int(cid), []).append(feat)

    return list(groups.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(threshold: float = 0.90) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    logging.info("Loading model input ...")
    df_raw     = pd.read_parquet(MODEL_INPUT, columns=FEATURE_COLS + ["label_k17"])
    labels_arr = df_raw["label_k17"].values
    features   = [f for f in FEATURE_COLS if f in df_raw.columns]
    logging.info("Features available: %d", len(features))

    # --- Rank-INT (parallel) ---
    logging.info("Applying rank-INT to %d features (parallel, %d workers) ...",
                 len(features), N_JOBS)
    int_vals = rank_int_df(df_raw, features)

    # --- η² for all features (parallel) ---
    logging.info("Computing η² ...")
    eta2 = eta_squared_all(int_vals, labels_arr, features)

    # --- Spearman correlation matrix ---
    # rank-INT output is already ranked → Pearson on rank-INT ≈ Spearman on raw
    logging.info("Computing Spearman correlation matrix (%d × %d) ...",
                 len(features), len(features))
    df_int_df = pd.DataFrame(int_vals)
    corr_abs  = df_int_df.corr(method="pearson").abs()   # fast; equivalent on rank-INT data

    # --- Complete-linkage grouping ---
    logging.info(
        "Building complete-linkage collinearity groups (|ρ| ≥ %.2f) ...", threshold
    )
    raw_groups = build_collinearity_groups(corr_abs, features, threshold)

    # Sort members within each group by η² desc, groups by their top-1 η²
    groups = sorted(
        [sorted(grp, key=lambda f: -eta2[f]) for grp in raw_groups],
        key=lambda grp: -eta2[grp[0]],
    )

    n_multi     = sum(1 for g in groups if len(g) > 1)
    n_singleton = sum(1 for g in groups if len(g) == 1)
    logging.info(
        "Found %d groups total: %d multi-feature, %d singletons.",
        len(groups), n_multi, n_singleton,
    )

    # --- Build output rows (top-1 per group retained) ---
    multi_gid  = 0
    group_rows = []

    for gid, grp in enumerate(groups, 1):
        top1_feat = grp[0]   # highest η² within group

        if len(grp) > 1:
            multi_gid += 1
            label    = f"Group {multi_gid} ({top1_feat})"
            dim_name = top1_feat
        else:
            label    = "Singleton"
            dim_name = top1_feat

        for rank, feat in enumerate(grp, 1):
            group_rows.append({
                "feature":         feat,
                "group_id":        gid,
                "group_size":      len(grp),
                "intragroup_rank": rank,
                "selected":        rank == 1,   # only top-1 retained
                "eta2":            round(eta2[feat], 6),
                "dimension_label": label,
                "dimension_name":  dim_name,
            })

    df_groups = (
        pd.DataFrame(group_rows)
        .sort_values(["group_id", "intragroup_rank"])
        .reset_index(drop=True)
    )
    df_groups.to_csv(OUT_TABLES / "feature_groups.csv", index=False)
    logging.info("Feature groups saved: %s", OUT_TABLES / "feature_groups.csv")

    # selected_features.csv — one row per group representative
    df_selected = (
        df_groups[df_groups["selected"]]
        .sort_values(["group_id", "intragroup_rank"])
        .reset_index(drop=True)
    )
    df_selected.to_csv(OUT_TABLES / "selected_features.csv", index=False)
    logging.info(
        "Selected %d representative features (removed %d redundant) → %s",
        len(df_selected),
        len(features) - len(df_selected),
        OUT_TABLES / "selected_features.csv",
    )

    # dimension_labels.json — full feature → dimension_label mapping for downstream
    feature_label_map: dict[str, str] = {
        row["feature"]: row["dimension_label"] for row in group_rows
    }
    labels_path = OUT_TABLES / "dimension_labels.json"
    labels_path.write_text(json.dumps(feature_label_map, indent=2, ensure_ascii=False))
    logging.info("Dimension labels saved: %s", labels_path)

    # Summary
    print(f"\n  Total features      : {len(features)}")
    print(f"  Collinear groups    : {n_multi}")
    print(f"  Singleton features  : {n_singleton}")
    print(f"  Representative feats: {len(df_selected)}  (one per group)")
    print(f"  Removed (redundant) : {len(features) - len(df_selected)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Complete-linkage collinearity grouping and feature deduplication.\n"
            "Retains the single top-η² representative per collinear group."
        )
    )
    parser.add_argument(
        "--threshold", type=float, default=0.90,
        help="Spearman |ρ| cutoff for grouping (default: 0.90).",
    )
    args = parser.parse_args()
    main(threshold=args.threshold)
