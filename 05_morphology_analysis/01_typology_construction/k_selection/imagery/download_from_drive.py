"""
Step 2 of 2-step satellite imagery pipeline.

Downloads completed GEE exports from Google Drive, then renders each cell
(satellite + buildings + roads + hex mask) into the final PNG.

Run after export_to_drive.py tasks are COMPLETED.

Usage:
    python imagery/download_from_drive.py --project YOUR_GEE_PROJECT_ID

    # Only download GeoTIFFs without rendering (for debugging):
    python imagery/download_from_drive.py --project YOUR_GEE_PROJECT_ID --tiff-only
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image

ROOT       = Path(__file__).resolve().parents[1]
PAIRS_CSV  = ROOT / "outputs" / "split_pairs" / "split_pairs_k02_to_k30.csv"
OUT_IMAGES = ROOT / "outputs" / "images"
LOG_DIR    = ROOT / "logs"
TASK_JSON  = ROOT / "logs" / "export_tasks.json"
TIFF_CACHE = ROOT / "outputs" / "tiff_cache"   # downloaded GeoTIFFs
LOG_DIR.mkdir(parents=True, exist_ok=True)
TIFF_CACHE.mkdir(parents=True, exist_ok=True)

# ../../../../03_morphology_imagery/. Override with
# AFRICA_MORPHOLOGY_IMAGERY_ROOT if your copy lives elsewhere.
AFRICA_MORPHOLOGY_IMAGERY = Path(os.environ.get(
    "AFRICA_MORPHOLOGY_IMAGERY_ROOT",
    "../../../../03_morphology_imagery",
))
sys.path.insert(0, str(AFRICA_MORPHOLOGY_IMAGERY))
from render import render_integrated, clip_to_cell  # noqa: E402

# SHP_cleaned geometry and the per-city building/road-network intermediates,
# both produced by ../../../../02_morphology_metrics/. Override with
# AFRICA_SHP_CLEANED_ROOT / AFRICA_MORPHOLOGY_METRICS_DATA_ROOT if your
# copy lives elsewhere.
FULL_GRID_ROOT = Path(os.environ.get(
    "AFRICA_SHP_CLEANED_ROOT",
    "../../../../02_morphology_metrics/outputs/SHP_cleaned",
))
MOMEPY_DATA_ROOT = Path(os.environ.get(
    "AFRICA_MORPHOLOGY_METRICS_DATA_ROOT",
    "../../../../02_morphology_metrics/data",
))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "download_from_drive.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def _fs_name(name: str) -> str:
    return name.replace("/", "_")


_shp_cache: dict = {}
_bldg_cache: dict = {}
_road_cache: dict = {}
_CACHE_MAX = 3  # keep at most 3 cities in memory at once


def get_cell_row(grid_id, city, country):
    key = (country, city)
    if key not in _shp_cache:
        city_fs   = _fs_name(city)
        shp_dir   = FULL_GRID_ROOT / country / city_fs
        shp_files = list(shp_dir.glob("*.shp")) if shp_dir.exists() else []
        _shp_cache[key] = gpd.read_file(shp_files[0]) if shp_files else None
    gdf = _shp_cache[key]
    if gdf is None:
        return None
    row = gdf[gdf["GridID"] == grid_id]
    if row.empty:
        return None
    geom = row.geometry.iloc[0]
    crs  = gdf.crs.to_string()
    return geom, crs, geom.centroid.x, geom.centroid.y


def get_vector_layers(city, country, target_crs):
    key = (country, city)
    if key not in _bldg_cache:
        # Evict oldest entries if cache is too large
        while len(_bldg_cache) >= _CACHE_MAX:
            oldest = next(iter(_bldg_cache))
            del _bldg_cache[oldest]
            _road_cache.pop(oldest, None)
            _shp_cache.pop(oldest, None)

        city_fs   = _fs_name(city)
        city_dir  = MOMEPY_DATA_ROOT / country / city_fs
        bldg_path = city_dir / "Output" / "open_buildings.shp"
        road_path = city_dir / "highways" / f"{city_fs}_highways.geojson"

        if bldg_path.exists():
            b = gpd.read_file(bldg_path)
            _bldg_cache[key] = b.to_crs(target_crs) if b.crs.to_string() != target_crs else b
        else:
            _bldg_cache[key] = gpd.GeoDataFrame(geometry=[])

        if road_path.exists():
            r = gpd.read_file(road_path)
            _road_cache[key] = r.to_crs(target_crs) if r.crs.to_string() != target_crs else r
        else:
            _road_cache[key] = gpd.GeoDataFrame(geometry=[])

    return _bldg_cache[key], _road_cache[key]


def build_drive_index(drive_service, folder_id: str) -> dict:
    """One-shot: fetch all file IDs in the Drive folder. Returns {grid_id: file_id}."""
    log.info("Building Drive file index ...")
    index = {}
    page_token = None
    while True:
        r = drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name)",
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        for f in r.get("files", []):
            gid = f["name"].replace(".tif", "")
            index[gid] = f["id"]
        page_token = r.get("nextPageToken")
        if not page_token:
            break
    log.info(f"  Drive index: {len(index)} files")
    return index


def download_tiff_from_drive(grid_id: str, file_id: str, creds) -> Path | None:
    """Download one GeoTIFF via requests (more stable than MediaIoBaseDownload)."""
    import time, requests
    tiff_path = TIFF_CACHE / f"{grid_id}.tif"
    if tiff_path.exists():
        return tiff_path
    if file_id is None:
        log.warning(f"Not found in Drive index: {grid_id}")
        return None
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    for attempt in range(1, 4):
        try:
            from google.auth.transport.requests import Request as GRequest
            creds.refresh(GRequest())
            headers = {"Authorization": f"Bearer {creds.token}"}
            r = requests.get(url, headers=headers, timeout=120, stream=True)
            r.raise_for_status()
            with open(tiff_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            return tiff_path
        except Exception as e:
            tiff_path.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(3 * attempt)
            else:
                log.warning(f"Download failed {grid_id}: {e}")
    return None


def render_cell(grid_id, city, country, tiff_path):
    """Render final PNG from downloaded GeoTIFF + vector overlays."""
    out_path = OUT_IMAGES / _fs_name(country) / _fs_name(city) / f"{grid_id}.png"
    if out_path.exists():
        return True, "cached"

    result = get_cell_row(grid_id, city, country)
    if result is None:
        return False, "geometry not found"
    cell_geom, crs, cx, cy = result

    buildings, highways = get_vector_layers(city, country, crs)
    bldg_clip = clip_to_cell(buildings, cell_geom)
    road_clip  = clip_to_cell(highways,  cell_geom)

    # Load satellite raster from GeoTIFF
    sat_img, sat_bounds = None, None
    if tiff_path and tiff_path.exists():
        import rasterio
        from rasterio.warp import reproject, Resampling, calculate_default_transform
        try:
            with rasterio.open(tiff_path) as src:
                # Reproject to cell CRS if needed
                if src.crs.to_string() != crs:
                    transform, width, height = calculate_default_transform(
                        src.crs, crs, src.width, src.height, *src.bounds)
                    data = np.zeros((3, height, width), dtype=np.uint8)
                    for i in range(1, 4):
                        reproject(source=rasterio.band(src, i), destination=data[i-1],
                                  src_transform=src.transform, src_crs=src.crs,
                                  dst_transform=transform, dst_crs=crs,
                                  resampling=Resampling.bilinear)
                    bounds = rasterio.transform.array_bounds(height, width, transform)
                else:
                    data   = src.read()[:3]
                    bounds = src.bounds
                    bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
                rgb     = np.transpose(data, (1, 2, 0))
                sat_img = Image.fromarray(rgb.astype(np.uint8), "RGB")
                sat_bounds = bounds
        except Exception as e:
            log.warning(f"Corrupt TIFF {grid_id}, deleting for re-download: {e}")
            tiff_path.unlink(missing_ok=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_integrated(cell_geom, bldg_clip, road_clip, sat_img, sat_bounds, out_path, crs)
    return True, "ok"


DRIVE_FOLDER_ID = None  # resolved at runtime from folder name


def get_drive_service():
    """Build Google Drive API service using GEE's stored OAuth credentials."""
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    import json, os

    # GEE stores credentials at ~/.config/earthengine/credentials
    cred_path = os.path.expanduser("~/.config/earthengine/credentials")
    with open(cred_path) as f:
        raw = json.load(f)

    from google.auth.transport.requests import Request
    creds = Credentials(
        token=None,
        refresh_token=raw.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=raw.get("client_id"),
        client_secret=raw.get("client_secret"),
        scopes=raw.get("scopes"),
    )
    creds.refresh(Request())
    service = build("drive", "v3", credentials=creds)
    return service, creds


def resolve_drive_folder(drive_service, folder_name: str) -> str:
    """Return Drive folder ID for the given folder name."""
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])
    if not files:
        raise RuntimeError(f"Drive folder '{folder_name}' not found. "
                           "GEE exports may not have started yet, or folder name differs.")
    return files[0]["id"]


def main():
    global DRIVE_FOLDER_ID, TASK_JSON, OUT_IMAGES, TIFF_CACHE

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--tiff-only", action="store_true",
                        help="Only download GeoTIFFs, skip rendering")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel download workers (default: 16)")
    parser.add_argument("--shard", type=int, default=0,
                        help="Shard index (0-based) for multi-process parallelism")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards")
    parser.add_argument("--task-json", type=Path, default=None,
                        help="Override TASK_JSON path")
    parser.add_argument("--out-images", type=Path, default=None,
                        help="Override OUT_IMAGES path")
    parser.add_argument("--tiff-cache", type=Path, default=None,
                        help="Override TIFF_CACHE path")
    parser.add_argument("--drive-folder", type=str, default="GEE_K_Selection_Exports",
                        help="Google Drive folder name")
    args = parser.parse_args()

    if args.task_json:  TASK_JSON  = args.task_json
    if args.out_images:
        OUT_IMAGES = args.out_images
        OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    if args.tiff_cache:
        TIFF_CACHE = args.tiff_cache
        TIFF_CACHE.mkdir(parents=True, exist_ok=True)

    ee.Initialize(project=args.project)
    log.info(f"Earth Engine initialized (project={args.project})")

    drive_svc, drive_creds = get_drive_service()
    DRIVE_FOLDER_ID = resolve_drive_folder(drive_svc, args.drive_folder)
    log.info(f"Drive folder resolved: {DRIVE_FOLDER_ID}")

    tasks = json.loads(TASK_JSON.read_text()) if TASK_JSON.exists() else {}
    if not tasks:
        log.error("No task list found. Run export_to_drive.py first.")
        sys.exit(1)

    # One-shot Drive index: {grid_id: file_id}
    drive_index = build_drive_index(drive_svc, DRIVE_FOLDER_ID)

    # Build todo list (skip already-rendered)
    todo = []
    skipped = 0
    for grid_id, info in tasks.items():
        out_path = OUT_IMAGES / _fs_name(info["country"]) / _fs_name(info["city"]) / f"{grid_id}.png"
        if out_path.exists():
            skipped += 1
        else:
            todo.append((grid_id, info["city"], info["country"], drive_index.get(grid_id)))

    # Shard todo list for multi-process parallelism
    if args.num_shards > 1:
        todo = [t for i, t in enumerate(todo) if i % args.num_shards == args.shard]
        log.info(f"Shard {args.shard}/{args.num_shards}: processing {len(todo)} items")

    total = len(tasks)
    log.info(f"Total={total}  already done={skipped}  to process={len(todo)}")

    if not todo:
        log.info("All done!")
        return

    # Producer/consumer pipeline:
    # - N download threads fetch GeoTIFFs from Drive (thread-safe)
    # - 1 render thread does matplotlib (not thread-safe)
    import queue as _queue
    import threading as _threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    N_DOWNLOAD_WORKERS = getattr(args, 'workers', 4)
    render_queue = _queue.Queue(maxsize=N_DOWNLOAD_WORKERS * 2)
    success = failed = 0
    done_count = [0]

    def download_worker(item):
        grid_id, city, country, file_id = item
        tiff_path = download_tiff_from_drive(grid_id, file_id, drive_creds)
        return grid_id, city, country, tiff_path

    def producer():
        with ThreadPoolExecutor(max_workers=N_DOWNLOAD_WORKERS) as pool:
            futs = {pool.submit(download_worker, item): item[0] for item in todo}
            for fut in as_completed(futs):
                render_queue.put(fut.result())
        render_queue.put(None)  # sentinel

    prod_thread = _threading.Thread(target=producer, daemon=True)
    prod_thread.start()

    while True:
        item = render_queue.get()
        if item is None:
            break
        grid_id, city, country, tiff_path = item
        ok, msg = render_cell(grid_id, city, country, tiff_path)
        done_count[0] += 1
        if ok:
            success += 1
        else:
            failed += 1
            log.warning(f"FAILED {grid_id}: {msg}")
        if done_count[0] % 50 == 0:
            log.info(f"Progress: {done_count[0]}/{len(todo)}  success={success}  failed={failed}")

    prod_thread.join()
    log.info(f"\nDone. success={success}  skipped={skipped}  failed={failed}  total={total}")


if __name__ == "__main__":
    main()
