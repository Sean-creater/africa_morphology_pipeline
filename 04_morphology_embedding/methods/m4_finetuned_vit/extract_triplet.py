"""
M4 (Triplet): Extract embeddings from triplet-trained projection head.

Mirrors extract.py but loads best_projection_head_triplet.pth.
Output: outputs/m4_finetuned_vit/embeddings/all_embeddings_triplet.npz
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    WEIGHTS_PATH, PROJECTION_DIM, DEVICE, IMG_SIZE,
    BASELINE_DIR, EMBEDDINGS_DIR,
)
from model import MetricModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.paths import ANALYSIS_IMAGES_DIR, VAL_PAIRS_IMAGES_DIR

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _run_batch(model, transform, images, gid_fn, device):
    tensors, gids = [], []
    for p in images:
        try:
            tensors.append(transform(Image.open(p).convert("RGB")))
            gids.append(gid_fn(p))
        except Exception as e:
            print(f"    Error {p.name}: {e}")
    if not tensors:
        return [], []
    with torch.no_grad():
        z = model(torch.stack(tensors).to(device)).cpu().numpy()
    return gids, list(z)


def extract():
    print("=" * 70)
    print("M4 (Triplet): Extracting embeddings (256-dim)")
    print("=" * 70)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    model = MetricModel(str(WEIGHTS_PATH), projection_dim=PROJECTION_DIM)
    model.head.load_state_dict(
        torch.load(BASELINE_DIR / "best_projection_head_triplet.pth", map_location="cpu")
    )
    model = model.to(DEVICE).eval()
    print("Triplet model loaded.")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        NORMALIZE,
    ])

    all_embeddings, all_grid_ids = [], []
    seen_gids = set()

    # ── 1. Analysis set ────────────────────────────────────────────────────
    cities = sorted(d for d in ANALYSIS_IMAGES_DIR.iterdir() if d.is_dir())
    print(f"\nProcessing {len(cities)} analysis-set cities...")

    for city_dir in tqdm(cities, desc="Cities"):
        images = sorted(city_dir.glob("*.png"))
        if not images:
            continue
        for i in range(0, len(images), 32):
            gids, embs = _run_batch(model, transform, images[i:i+32],
                                    lambda p: "_".join(p.stem.split("_")[:-1]),
                                    DEVICE)
            for gid, e in zip(gids, embs):
                if gid not in seen_gids:
                    all_embeddings.append(e)
                    all_grid_ids.append(gid)
                    seen_gids.add(gid)

    # ── 2. Val pairs ───────────────────────────────────────────────────────
    val_images = sorted(VAL_PAIRS_IMAGES_DIR.rglob("*.png"), key=lambda p: p.name)
    print(f"\nProcessing {len(val_images)} val-pair images...")

    def _val_gid(p):
        return p.stem.lstrip("AB").lstrip("_")

    for i in tqdm(range(0, len(val_images), 32), desc="Val pairs"):
        gids, embs = _run_batch(model, transform, val_images[i:i+32],
                                _val_gid, DEVICE)
        for gid, e in zip(gids, embs):
            if gid not in seen_gids:
                all_embeddings.append(e)
                all_grid_ids.append(gid)
                seen_gids.add(gid)

    all_emb_arr = np.array(all_embeddings)
    out_path = EMBEDDINGS_DIR / "all_embeddings_triplet.npz"
    np.savez_compressed(out_path,
                        embeddings=all_emb_arr,
                        grid_ids=np.array(all_grid_ids))
    print(f"\nTotal: {len(all_emb_arr)} embeddings {all_emb_arr.shape}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    extract()
