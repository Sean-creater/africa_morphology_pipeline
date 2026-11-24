# K Selection

Materials supporting the choice of k = 17 morphological types, used
throughout this repository and the companion analysis modules: sample
generation, image rendering, and the resulting evaluation scores across
candidate k = 2–30.

## Method

1. `sampling/build_dendrogram_and_sample.py` — Mini-batch K-Means
   compression of the full-Africa embeddings to 2,000 centroids, Ward HAC
   on the centroids, and sampling of cross-boundary cell pairs at each
   candidate split k = 2–30 (drawn from the full ~1M-cell pool, not just
   the centroids).
2. `imagery/` — renders each sampled cell into the same satellite +
   building/road overlay composite used throughout this project (via
   `../../../03_morphology_imagery/`), and exports/downloads the
   underlying satellite tiles through Google Earth Engine.
3. `within_cluster/` — a second sampling pass drawing within-cluster pairs
   at each k, used to correct a systematic bias in a cross-boundary-only
   score: at low k, clusters are large and heterogeneous, so cross-boundary
   pairs are judged "different" almost by default, inflating low-k scores.
   Combining both gives a discrimination score,
   `score(k) = P(different | cross-boundary) − P(different | within-cluster)`,
   which is less biased toward low k.
4. `annotation_results/` — the resulting evaluation, combining pairwise
   judgments across both sampling passes into per-k scores under several
   aggregation schemes (vote-level average, majority vote, and confidence-
   weighted average), plus a leave-one-batch-out stability check across
   independent annotation rounds.

k = 17 was selected from this evaluation together with the diagnostic
metrics in `../typology_construction/03_kselection_diagnostics.py`, and is
used as the working number of morphological types throughout this
repository. See the manuscript's Methods and Supplementary Information for
the full selection procedure.

## Data requirements

- Continent-scale student embeddings, produced by
  `../../../04_morphology_embedding/`.
- Vector building/road layers and grid geometries, produced by
  `../../../02_morphology_metrics/`.
- `../../../03_morphology_imagery/` for image rendering
  (`AFRICA_MORPHOLOGY_IMAGERY` in `imagery/generate_images.py` and
  `imagery/download_from_drive.py`).
- A Google Earth Engine project for satellite tile access.

## Usage

```bash
python sampling/build_dendrogram_and_sample.py

# Authenticate Earth Engine once, interactively:
python -c "import ee; ee.Authenticate()"

python imagery/generate_images.py --project YOUR_GEE_PROJECT_ID

# Within-cluster correction pass:
python within_cluster/sample_within_pairs.py
bash within_cluster/run_generate_images.sh
```

## annotation_results/

- `expert_k_evaluation/` — pairwise cross-boundary and within-cluster
  judgments (`*_expert_annotations.csv`), one row per rated pair.
  `Annotator_IP` is anonymised to `annotator_1`…`annotator_5`.
- `optimal_k/` — `k_evaluation_scores.csv`: per-k scores under each
  aggregation scheme, with the resulting rank of each candidate k;
  `elbow_metrics_k2_k30.csv` / `elbow_curve_k2_k30.svg`: a complementary
  elbow-style view of the same scores.
- `expert_k_evaluation_4choose3*/` — a leave-one-batch-out stability check:
  each of the 4 independent annotation batches is held out in turn and the
  best-k selection is recomputed on the remaining 3, under 4 scoring
  variants (confidence-weighted / majority-vote, each with a product or
  harmonic-mean combination of the separation and compactness terms). k=17
  is not the single best k under every variant and held-out batch, but it
  consistently ranks among the top candidates, which is the basis for
  treating it as a stable, if not uniquely optimal, choice within this
  range.
- `k_stability_check/` — a complementary check at the level of the
  resulting typology rather than the pairwise scores: cluster
  interpretability and cluster-count stability compared across k = 8, 17,
  and 23 (`typology_k08/`, `typology_k17/`, `typology_k23/`), using
  permutation-importance feature ranking
  (`feature_definitions_and_permutation_importance.csv`) rather than the
  rank-INT method used in `../cluster_interpretation/`.
