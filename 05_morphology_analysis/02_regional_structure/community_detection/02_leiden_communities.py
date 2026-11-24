"""
02_leiden_communities.py
========================
Build a continent-level city similarity graph from city feature vectors
(01_build_city_features.py) and detect communities with the Leiden algorithm.

City-to-city distance
---------------------
D(c, c') = α · D_P(c, c') + (1 − α) · D_A(c, c')

D_P : Jensen–Shannon distance on composition vectors p (17-dim).
      Bounded [0, 1], well-defined for sparse probability vectors.

D_A : Configuration distance on log-enrichment profiles A.
      For each city pair, only the intersection of *defined* A_ij pairs
      (O_ij > 0 in both cities) enters the Pearson correlation:

          r_cc' = Pearson(A_c[shared], A_c'[shared])
          D_A(c, c') = (1 − r_cc') / 2   ∈ [0, 1]

      Pairs with fewer than MIN_SHARED_PAIRS defined entries fall back
      to composition distance only (α overridden to 1.0 for that pair).

Similarity graph
----------------
k-NN graph on D.
Edge weight = exp(−D / σ),  σ = median of all pairwise distances.

Community detection
-------------------
Leiden, ModularityVertexPartition, weighted edges.
Best partition over N_SEEDS random seeds retained.
Sensitivity: α ∈ {0, 0.25, 0.5, 0.75, 1.0}  ×  k ∈ {10, 15, 20, 50, 100}.

Distance matrix caching
-----------------------
D_P and D_A are expensive to compute (~5 min for 6918 cities).
They are saved as float32 .npy files in outputs/cache/ keyed by a
hash of the input feature parquet.  On subsequent runs the cache is
loaded instantly, making k / α sweeps essentially free.

Cache is invalidated automatically whenever city_features.parquet changes.

Outputs
-------
outputs/city_communities.parquet          main result (α=0.5, k=100)
outputs/community_summary.parquet         per-community statistics
outputs/sensitivity/city_communities_alpha{a}_k{k}.parquet
outputs/shared_pairs_stats.parquet        diagnostic
outputs/cache/D_p_<hash>.npy             cached JSD distance matrix
outputs/cache/D_a_<hash>.npy             cached Pearson distance matrix

Usage
-----
    python 02_leiden_communities.py
    python 02_leiden_communities.py --alpha 0.5 --k 100 --n-seeds 10 --n-workers 16
    python 02_leiden_communities.py --skip-sensitivity   # main run only
"""

import argparse
import hashlib
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")

ROOT      = Path(__file__).parent
OUT_DIR   = ROOT / "outputs"
CACHE_DIR = OUT_DIR / "cache"
(OUT_DIR / "sensitivity").mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_TYPES          = 17
MIN_SHARED_PAIRS = 10

P_COLS = [f"p_{i}" for i in range(1, N_TYPES + 1)]
A_COLS = [f"A_{i}_{j}"
          for i in range(1, N_TYPES + 1)
          for j in range(i, N_TYPES + 1)]


# ---------------------------------------------------------------------------
# Distance helpers (module-level for pickling in multiprocessing)
# ---------------------------------------------------------------------------

def _js_row(args):
    """Compute one row of the JSD matrix."""
    i, P = args
    n = len(P)
    row = np.zeros(n, dtype=np.float32)
    for j in range(i + 1, n):
        row[j] = jensenshannon(P[i], P[j])
    return i, row


def composition_distances(P: np.ndarray, n_workers: int) -> np.ndarray:
    """Jensen–Shannon distance matrix using parallel row computation."""
    n = len(P)
    D = np.zeros((n, n), dtype=np.float32)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_js_row, (i, P)): i for i in range(n)}
        done = 0
        for fut in as_completed(futs):
            i, row = fut.result()
            D[i, i + 1:] = row[i + 1:]
            done += 1
            if done % 500 == 0:
                logging.info("  D_P rows done: %d / %d", done, n)
    D = D + D.T   # symmetrise
    return D


def _da_row(args):
    """Compute one row of the D_A matrix."""
    i, A, defined, min_shared = args
    n = len(A)
    row = np.full(n, np.nan, dtype=np.float32)
    for j in range(i + 1, n):
        shared = defined[i] & defined[j]
        if shared.sum() < min_shared:
            continue
        ai, aj = A[i, shared], A[j, shared]
        if ai.std() < 1e-12 or aj.std() < 1e-12:
            continue
        r = float(np.corrcoef(ai, aj)[0, 1])
        row[j] = np.clip((1.0 - r) / 2.0, 0.0, 1.0)
    return i, row


def configuration_distances(A: np.ndarray, n_workers: int) -> np.ndarray:
    """Pearson-based configuration distance matrix, parallel."""
    n = len(A)
    defined = ~np.isnan(A)
    D = np.full((n, n), np.nan, dtype=np.float32)
    np.fill_diagonal(D, 0.0)

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_da_row, (i, A, defined, MIN_SHARED_PAIRS)): i
                for i in range(n)}
        done = 0
        for fut in as_completed(futs):
            i, row = fut.result()
            D[i, i + 1:] = row[i + 1:]
            done += 1
            if done % 500 == 0:
                logging.info("  D_A rows done: %d / %d", done, n)

    # Symmetrise (NaN stays NaN)
    upper = ~np.isnan(D)
    D[np.tril_indices(n, -1)] = np.where(
        upper.T[np.tril_indices(n, -1)],
        D.T[np.tril_indices(n, -1)],
        np.nan)
    return D


def combined_distances(D_p: np.ndarray, D_a: np.ndarray,
                       alpha: float) -> np.ndarray:
    """α·D_P + (1−α)·D_A; fall back to D_P where D_A is undefined."""
    missing = np.isnan(D_a)
    D = alpha * D_p + (1.0 - alpha) * np.where(missing, 0.0, D_a)
    D[missing] = D_p[missing]
    return D.astype(np.float32)


# ---------------------------------------------------------------------------
# Distance matrix cache
# ---------------------------------------------------------------------------

def _features_hash(feat_path: Path) -> str:
    """MD5 of the feature parquet (stat-based: size + mtime, fast)."""
    stat = feat_path.stat()
    key  = f"{stat.st_size}_{stat.st_mtime_ns}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def load_or_compute_distances(feat_path: Path, P: np.ndarray,
                               A: np.ndarray, n_workers: int):
    """
    Load D_P and D_A from cache if available and fresh, otherwise compute
    and save them.  Cache files are keyed by a hash of the feature parquet
    so they are invalidated automatically when the input changes.

    Returns (D_p, D_a).
    """
    h = _features_hash(feat_path)
    dp_path = CACHE_DIR / f"D_p_{h}.npy"
    da_path = CACHE_DIR / f"D_a_{h}.npy"

    if dp_path.exists() and da_path.exists():
        logging.info("Loading cached distance matrices (hash=%s) …", h)
        D_p = np.load(dp_path)
        D_a = np.load(da_path)
        logging.info("  D_P shape=%s  D_A shape=%s", D_p.shape, D_a.shape)
    else:
        # Clean up any stale cache files from previous feature versions
        for old in CACHE_DIR.glob("D_p_*.npy"):
            old.unlink()
            logging.info("Removed stale cache: %s", old.name)
        for old in CACHE_DIR.glob("D_a_*.npy"):
            old.unlink()

        logging.info("Computing D_P (Jensen–Shannon, %d workers) …", n_workers)
        D_p = composition_distances(P, n_workers)
        np.save(dp_path, D_p)
        logging.info("  D_P saved → %s", dp_path.name)

        logging.info("Computing D_A (Pearson on defined pairs, %d workers) …",
                     n_workers)
        D_a = configuration_distances(A, n_workers)
        np.save(da_path, D_a)
        logging.info("  D_A saved → %s", da_path.name)

    return D_p, D_a


# ---------------------------------------------------------------------------
# k-NN graph
# ---------------------------------------------------------------------------

def build_knn_graph(D: np.ndarray, k: int) -> ig.Graph:
    """Symmetric (union) k-NN graph: city i and city j are connected if
    either one lists the other among its own k nearest neighbours.

    Keeping only edges with i < j (as an earlier version of this function
    did) silently drops any pair where the larger-indexed city selects the
    smaller-indexed one as a neighbour but not vice versa, making the graph
    depend on row order rather than purely on distance. Union
    symmetrization keeps every edge either side actively selected, with no
    duplicate edge when the selection is mutual.

    Self is excluded explicitly (row[i]=inf) rather than by assuming
    argsort places the zero self-distance at position 0 — some city pairs
    have identical 17-type composition (distance exactly 0), so another
    city can tie with self for the nearest position and argsort's ordering
    among ties is not something to rely on. Ties at the k-th neighbour are
    all kept (rather than arbitrarily broken by index), so a city's actual
    degree can exceed k when its k-th and (k+1)-th distances are equal.
    """
    n = len(D)
    if not np.allclose(D, D.T, atol=1e-8):
        raise ValueError("build_knn_graph: distance matrix D is not symmetric.")
    if not np.isfinite(D).all():
        raise ValueError("build_knn_graph: distance matrix D contains non-finite values.")
    if not np.allclose(np.diag(D), 0):
        raise ValueError("build_knn_graph: distance matrix D has non-zero diagonal.")
    if not (1 <= k < n):
        raise ValueError(f"build_knn_graph: k={k} must satisfy 1 <= k < n={n}.")

    sigma = float(np.median(D[D > 0]))
    directed_pairs = set()
    for i in range(n):
        row = D[i].copy()
        row[i] = np.inf
        kth_dist = np.partition(row, k - 1)[k - 1]
        nbrs = np.where(row <= kth_dist)[0]
        for j in nbrs:
            directed_pairs.add((int(i), int(j)))

    undirected_pairs = sorted({(min(i, j), max(i, j)) for i, j in directed_pairs})

    edges, weights = [], []
    for i, j in undirected_pairs:
        w = float(np.exp(-D[i, j] / sigma))
        edges.append((i, j))
        weights.append(w)

    G = ig.Graph(n=n, edges=edges, directed=False)
    G.es["weight"] = weights

    _validate_knn_graph(G, D, directed_pairs, k)

    degrees = G.degree()
    components = G.connected_components()
    giant_frac = max(components.sizes()) / n if n > 0 else 0.0
    logging.info(
        "k-NN graph (k=%d): %d nodes, %d edges, mean degree=%.1f, "
        "%d connected components, giant component=%.1f%%",
        k, n, G.ecount(), float(np.mean(degrees)), len(components), giant_frac * 100,
    )
    return G


def _validate_knn_graph(G: ig.Graph, D: np.ndarray, directed_pairs: set, k: int) -> None:
    """Structural checks on the symmetrized k-NN graph."""
    edge_pairs = [tuple(sorted(e.tuple)) for e in G.es]

    if any(i == j for i, j in edge_pairs):
        raise ValueError("k-NN graph contains self-loops.")
    if len(edge_pairs) != len(set(edge_pairs)):
        raise ValueError("k-NN graph contains duplicate edges.")

    # Every city must retain at least one edge to each city it actively
    # selected as a neighbour (the union step can only add edges relative
    # to any single city's own directed selection, never remove one).
    edge_set = set(edge_pairs)
    for i, j in directed_pairs:
        if (min(i, j), max(i, j)) not in edge_set:
            raise ValueError(
                f"k-NN graph is missing edge ({i}, {j}), which city {i} "
                "actively selected as a k-nearest neighbour."
            )


# ---------------------------------------------------------------------------
# Leiden
# ---------------------------------------------------------------------------

def run_leiden(G: ig.Graph, n_seeds: int) -> list:
    best_part, best_q = None, -np.inf
    for seed in range(n_seeds):
        part = leidenalg.find_partition(
            G, leidenalg.ModularityVertexPartition,
            weights="weight", seed=seed)
        if part.modularity > best_q:
            best_q, best_part = part.modularity, part
    logging.info("Leiden Q=%.4f  communities=%d  (best of %d seeds)",
                 best_q, len(best_part), n_seeds)
    return list(best_part.membership)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_communities(df: pd.DataFrame, membership: list,
                     alpha: float, k: int, main: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["community"] = membership
    cols = ["country", "city", "n_cells", "n_types", "community"]
    suffix = f"alpha{str(alpha).replace('.','')}_k{k}"
    if main:
        df[cols].to_parquet(OUT_DIR / "city_communities.parquet", index=False)
    df[cols].to_parquet(OUT_DIR / "sensitivity" / f"city_communities_{suffix}.parquet",
                        index=False)
    return df


def save_summary(df: pd.DataFrame, alpha: float, k: int) -> None:
    rows = []
    for comm_id, grp in df.groupby("community"):
        row = {"community": int(comm_id),
               "n_cities":  len(grp),
               "n_cells":   int(grp["n_cells"].sum()),
               "top_country": grp["country"].value_counts().index[0]}
        for col in P_COLS:
            row[f"mean_{col}"] = float(grp[col].mean())
        rows.append(row)
    summary = (pd.DataFrame(rows)
               .sort_values("community").reset_index(drop=True))
    summary.to_parquet(OUT_DIR / "community_summary.parquet", index=False)
    logging.info("Saved community summary → %s", OUT_DIR / "community_summary.parquet")
    print("\nCommunity overview (main α=%.2f, k=%d):" % (alpha, k))
    print(summary[["community", "n_cities", "n_cells", "top_country"]]
          .to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",            type=float, default=0.5)
    parser.add_argument("--k",                type=int,   default=100)
    parser.add_argument("--n-seeds",          type=int,   default=10)
    parser.add_argument("--n-workers",        type=int,   default=16)
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="Run main α/k only (skip sensitivity sweep)")
    args = parser.parse_args()

    feat_path = OUT_DIR / "city_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"{feat_path} not found — run 01_build_city_features.py first.")

    df = pd.read_parquet(feat_path)
    logging.info("Loaded %d cities.", len(df))

    P = df[P_COLS].values.astype(np.float32)
    A = df[A_COLS].values.astype(np.float32)

    # --- Distance matrices (cached) ---
    D_p, D_a = load_or_compute_distances(feat_path, P, A, args.n_workers)

    n_def   = int((~np.isnan(D_a)).sum() // 2)
    n_total = len(df) * (len(df) - 1) // 2
    logging.info("D_A defined for %d / %d pairs (%.1f%%)",
                 n_def, n_total, 100 * n_def / n_total)

    # Shared-pair diagnostic (only written once; skip if file already exists)
    sp_path = OUT_DIR / "shared_pairs_stats.parquet"
    if not sp_path.exists():
        defined_mask = ~np.isnan(A)
        shared = np.array([
            int((defined_mask[i] & defined_mask[j]).sum())
            for i in range(len(df)) for j in range(i + 1, len(df))
        ])
        pd.DataFrame({"shared_pairs": shared}).to_parquet(sp_path, index=False)
        logging.info("Shared defined pairs: median=%d  min=%d  max=%d",
                     int(np.median(shared)), int(shared.min()), int(shared.max()))

    # --- Main run ---
    logging.info("Main run: α=%.2f, k=%d …", args.alpha, args.k)
    D_main   = combined_distances(D_p, D_a, args.alpha)
    G_main   = build_knn_graph(D_main, k=args.k)
    mem_main = run_leiden(G_main, n_seeds=args.n_seeds)
    df_main  = save_communities(df, mem_main, args.alpha, args.k, main=True)

    # Print summary inline
    vc = pd.Series(mem_main).value_counts().sort_index()
    print(f"\nMain run (α={args.alpha}, k={args.k}): "
          f"{len(vc)} communities, Q stored in log above")
    print("Sizes:", vc.values.tolist())
    save_summary(df_main, args.alpha, args.k)

    if args.skip_sensitivity:
        logging.info("Sensitivity sweep skipped (--skip-sensitivity).")
        logging.info("All done.")
        return

    # --- Sensitivity: α × k ---
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    ks     = [10, 15, 20, 50, 100, 150, 200]
    logging.info("Sensitivity analysis: %d α values × %d k values …",
                 len(alphas), len(ks))
    for alpha_s in alphas:
        D_s = combined_distances(D_p, D_a, alpha_s)
        for k_s in ks:
            if alpha_s == args.alpha and k_s == args.k:
                continue   # already saved as main
            G_s   = build_knn_graph(D_s, k=k_s)
            mem_s = run_leiden(G_s, n_seeds=args.n_seeds)
            save_communities(df, mem_s, alpha_s, k_s)

    logging.info("All done.")


if __name__ == "__main__":
    main()
