"""
M3: Extract frozen DINOv3 embeddings (no training, no projection head).

Backbone output (1024-dim CLS token) is L2-normalized and saved directly.
This is the simplest baseline — zero supervision, zero fine-tuning.

Covers two image sets:
  1. analysis_set  — 5,494 annotated images (city sub-dirs)
  2. val_pairs     — 1,120 images from pairwise evaluation set
     (stored as {pair_folder}/A_xxx.png, B_xxx.png)

Output: outputs/m3_frozen_vit/embeddings/
  all_embeddings.npz  — embeddings [N, 1024], grid_ids [N]
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backbone" / "dinov3"))
from dinov3.hub.backbones import dinov3_vitl16

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.paths import WEIGHTS_PATH, ANALYSIS_IMAGES_DIR, VAL_PAIRS_IMAGES_DIR

sys.path.insert(0, str(Path(__file__).parent))
from config import DEVICE, IMG_SIZE, EMBEDDINGS_DIR


NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        NORMALIZE,
    ])


def _run_batch(backbone, transform, images, gid_fn, device):
    """Embed a list of image paths; returns (gids, embeddings)."""
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
        feats = backbone(torch.stack(tensors).to(device))
        feats = F.normalize(feats, dim=-1).cpu().numpy()
    return gids, list(feats)


def extract():
    print("=" * 70)
    print("M3: Extracting frozen DINOv3 embeddings (1024-dim, L2-normalized)")
    print("=" * 70)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    backbone = dinov3_vitl16(pretrained=True, weights=str(WEIGHTS_PATH))
    backbone = backbone.to(DEVICE).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print("Backbone loaded.")

    transform = get_transform()
    all_embeddings, all_grid_ids = [], []
    seen_gids = set()

    cities = sorted(d for d in ANALYSIS_IMAGES_DIR.iterdir() if d.is_dir())
    print(f"\nProcessing {len(cities)} analysis-set cities...")

    for city_dir in tqdm(cities, desc="Cities"):
        images = sorted(city_dir.glob("*.png"))
        if not images:
            continue
        city = city_dir.name
        out_dir = EMBEDDINGS_DIR / city
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_dict = {}
        for i in range(0, len(images), 32):
            gids, embs = _run_batch(backbone, transform, images[i:i+32],
                                    lambda p: "_".join(p.stem.split("_")[:-1]),
                                    DEVICE)
            for gid, e in zip(gids, embs):
                emb_dict[gid] = e
                if gid not in seen_gids:
                    all_embeddings.append(e)
                    all_grid_ids.append(gid)
                    seen_gids.add(gid)
        np.savez_compressed(out_dir / f"{city}_embeddings.npz", **emb_dict)

    val_images = sorted(VAL_PAIRS_IMAGES_DIR.rglob("*.png"), key=lambda p: p.name)
    print(f"\nProcessing {len(val_images)} val-pair images...")

    def _val_gid(p):
        return p.stem.lstrip("AB").lstrip("_")

    for i in tqdm(range(0, len(val_images), 32), desc="Val pairs"):
        gids, embs = _run_batch(backbone, transform, val_images[i:i+32],
                                _val_gid, DEVICE)
        for gid, e in zip(gids, embs):
            if gid not in seen_gids:
                all_embeddings.append(e)
                all_grid_ids.append(gid)
                seen_gids.add(gid)

    all_emb_arr = np.array(all_embeddings)
    np.savez_compressed(
        EMBEDDINGS_DIR / "all_embeddings.npz",
        embeddings=all_emb_arr,
        grid_ids=np.array(all_grid_ids),
    )

    print(f"\nTotal: {len(all_emb_arr)} embeddings {all_emb_arr.shape}")
    print(f"  (analysis set: {len(seen_gids) - len(val_images)} unique grid_ids "
          f"+ val pairs contribution)")
    print(f"Saved to: {EMBEDDINGS_DIR}")


if __name__ == "__main__":
    extract()
