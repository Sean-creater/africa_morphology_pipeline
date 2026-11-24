"""
Evaluate M1-M6 on:
  1. Pairwise PRA & Spearman rho  (535 high-consensus pairs: round2 + anchor)
  2. Triplet Accuracy             (546 triplets derived from anchor pairs)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.dataset import load_morphology_features
from shared.paths import FEATURES_CSV, FEATURES_VAL_CSV, OUTPUTS_DIR, EVAL_PAIRS_CSV, EVAL_TRIPLETS_CSV
from methods.m6_triplet_distillation.models import StudentMLP

pairs = pd.read_csv(EVAL_PAIRS_CSV)
trips = pd.read_csv(EVAL_TRIPLETS_CSV)
print(f"Eval pairs: {len(pairs)}  |  Triplets: {len(trips)}")

feat_a, gids_a, feature_names, _ = load_morphology_features(FEATURES_CSV)
feat_v, gids_v, _, _             = load_morphology_features(FEATURES_VAL_CSV)
existing = set(gids_a)
new_idx  = [i for i, g in enumerate(gids_v) if g not in existing]
features = np.vstack([feat_a, feat_v[new_idx]])
grid_ids = gids_a + [gids_v[i] for i in new_idx]


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    embs = d['embeddings'].astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.where(norms > 0, norms, 1.0)
    return {g: e for g, e in zip(d['grid_ids'], embs)}


def get_student_emb(model_dir):
    best = None
    for p in sorted(Path(model_dir).glob("student_fold*.pth")):
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if best is None or ckpt["best_val_loss"] < best[1]:
            best = (p, ckpt["best_val_loss"], ckpt)
    ckpt = best[2]
    model = StudentMLP(ckpt["input_dim"], ckpt["hidden_dims"],
                       ckpt["dropout"], ckpt["batch_norm"], ckpt["output_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    X = ckpt["scaler"].transform(features).clip(-10, 10).astype(np.float32)
    with torch.no_grad():
        embs = model(torch.from_numpy(X)).numpy()
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.where(norms > 0, norms, 1.0)
    return {g: embs[i] for i, g in enumerate(grid_ids)}


def evaluate(emb_map, name):
    sims, ratings = [], []
    for _, row in pairs.iterrows():
        ga, gb = row['gid_A'], row['gid_B']
        if ga in emb_map and gb in emb_map:
            sims.append(float(np.dot(emb_map[ga], emb_map[gb])))
            ratings.append(float(row['Rating']))
    s, r = np.array(sims), np.array(ratings)
    si = s[:,None]-s[None,:]; ri = r[:,None]-r[None,:]
    mask = ri != 0
    pra = float(((si*ri>0)&mask).sum()/mask.sum())
    rho, _ = stats.spearmanr(s, r)
    n_pairs = len(sims)

    correct, total = 0, 0
    for _, row in trips.iterrows():
        a, p, n = row['anchor'], row['positive'], row['negative']
        if a in emb_map and p in emb_map and n in emb_map:
            correct += int(np.dot(emb_map[a], emb_map[p]) >
                           np.dot(emb_map[a], emb_map[n]))
            total += 1
    ta = correct / total if total > 0 else 0

    print(f"  {name:<30s} | pairs={n_pairs} | PRA={pra*100:.2f}% | rho={rho:.4f} | TriAcc={ta*100:.2f}% ({total} triplets)")
    return {"Method": name, "N_pairs": n_pairs, "PRA": round(pra*100,2),
            "Spearman_rho": round(rho,4), "Triplet_Acc": round(ta*100,2),
            "N_triplets": total}


print("\n" + "="*85)
print(f"  {'Method':<30s} | pairs  | PRA     | rho    | TriAcc  (triplets)")
print("="*85)

results = []
BASE = OUTPUTS_DIR

results.append(evaluate(load_npz(BASE/"m1_raw_features/embeddings/all_embeddings_pca.npz"), "M1_PCA"))
results.append(evaluate(load_npz(BASE/"m3_frozen_vit/embeddings/all_embeddings.npz"),       "M3_FrozenViT"))
results.append(evaluate(load_npz(BASE/"m5_selfsup_distillation/embeddings/all_embeddings.npz"), "M5_SelfSupDistill"))
results.append(evaluate(load_npz(BASE/"m2_supervised_morphological/embeddings/all_embeddings_triplet.npz"), "M2_Triplet"))
results.append(evaluate(load_npz(BASE/"m4_finetuned_vit/embeddings/all_embeddings_triplet.npz"),            "M4_Triplet"))
results.append(evaluate(get_student_emb(BASE/"m6_triplet_distillation/models"),                             "M6_Triplet"))

print("="*85)

out_dir = BASE / "eval_results"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "final_ranking.csv"
pd.DataFrame(results).to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
