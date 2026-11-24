"""
check_knn_graph_row_order_invariance.py
=========================================
One-off robustness check for the symmetric k-NN graph construction in
02_leiden_communities.py: confirms that the graph itself — not just the
downstream Leiden partition — does not depend on the row order of the
input city feature table.

Method
------
1. Load the cached city distance matrix and build the main k-NN graph, as
   in 02_leiden_communities.py's main run.
2. Apply a random permutation to the city order (a row/column permutation
   of the same distance matrix — no distances are recomputed; the point of
   this check is that permuting rows and rebuilding the graph must recover
   the identical graph, not that recomputing distances from raw features
   would also agree, which is a separate and uninteresting property of the
   distance function itself).
3. Map the permuted graph's edges back to the original city indices and
   assert the edge set — and each edge's weight — is IDENTICAL to the
   original graph's, not merely similar. This is a direct structural
   check on build_knn_graph(), independent of anything Leiden does.
4. Only after that direct check passes, additionally run Leiden on both
   graphs and report the adjusted Rand index between the two partitions,
   as a diagnostic of algorithmic (seed/heuristic) stability — not as
   evidence of graph correctness, since Leiden's own optimisation
   heuristic can equally produce ARI < 1 between two IDENTICAL graphs run
   with different seeds. Step 3 is the actual correctness proof; step 4
   is supplementary.

Usage
-----
    python check_knn_graph_row_order_invariance.py
"""
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

lc = import_module("02_leiden_communities")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RANDOM_SEED = 42


def _edge_weight_map(G, index_map=None):
    """{(min(u,v), max(u,v)): weight}, with vertex ids remapped through
    index_map (permuted index -> original index) if given."""
    out = {}
    for e in G.es:
        u, v = e.tuple
        if index_map is not None:
            u, v = int(index_map[u]), int(index_map[v])
        key = (min(u, v), max(u, v))
        out[key] = float(e["weight"])
    return out


def main():
    feat_path = lc.OUT_DIR / "city_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} not found — run 01_build_city_features.py first.")

    import pandas as pd
    df = pd.read_csv(feat_path) if feat_path.suffix == ".csv" else pd.read_parquet(feat_path)
    P = df[lc.P_COLS].values.astype(np.float32)
    A = df[lc.A_COLS].values.astype(np.float32)

    log.info("Building original-order graph ...")
    D_p, D_a = lc.load_or_compute_distances(feat_path, P, A, n_workers=8)
    D_main = lc.combined_distances(D_p, D_a, alpha=0.5)
    G_main = lc.build_knn_graph(D_main, k=100)

    log.info("Building row-permuted graph (same distances, permuted rows/cols) ...")
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(df)
    perm = rng.permutation(n)          # perm[new_idx] = original_idx
    D_perm = D_main[np.ix_(perm, perm)]
    G_perm = lc.build_knn_graph(D_perm, k=100)

    # --- Step 3: direct structural comparison (the actual correctness proof) ---
    edges_main = _edge_weight_map(G_main)
    edges_perm_mapped = _edge_weight_map(G_perm, index_map=perm)

    same_keys = set(edges_main) == set(edges_perm_mapped)
    if not same_keys:
        only_main = set(edges_main) - set(edges_perm_mapped)
        only_perm = set(edges_perm_mapped) - set(edges_main)
        log.error(
            "FAIL: edge sets differ. %d edges only in original, %d only in "
            "permuted (mapped back). Examples: main=%s perm=%s",
            len(only_main), len(only_perm),
            list(only_main)[:5], list(only_perm)[:5],
        )
        raise SystemExit(1)

    max_weight_diff = max(
        abs(edges_main[k] - edges_perm_mapped[k]) for k in edges_main
    )
    if max_weight_diff > 1e-10:
        log.error("FAIL: edge sets match but weights differ, max abs diff = %.3e",
                   max_weight_diff)
        raise SystemExit(1)

    log.info(
        "PASS (structural): %d edges identical between original and "
        "permuted-then-remapped graphs, max weight diff = %.2e.",
        len(edges_main), max_weight_diff,
    )

    # --- Step 4: Leiden ARI as a supplementary algorithmic-stability diagnostic ---
    mem_main = lc.run_leiden(G_main, n_seeds=5)
    mem_perm = lc.run_leiden(G_perm, n_seeds=5)
    inv_perm = np.argsort(perm)
    mem_perm_original_order = np.asarray(mem_perm)[inv_perm]

    ari = adjusted_rand_score(mem_main, mem_perm_original_order)
    log.info(
        "Supplementary diagnostic — Leiden ARI (original vs. row-permuted): "
        "%.4f (this reflects Leiden's own optimisation stability given an "
        "already-verified-identical graph, not graph correctness).",
        ari,
    )


if __name__ == "__main__":
    main()
