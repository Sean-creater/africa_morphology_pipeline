"""
Step 1 of 2-step satellite imagery pipeline.

Submits one GEE Export.image.toDrive() task per grid cell in split_pairs_k02_to_k30.csv.
GEE processes all tasks in parallel in the cloud. When done, run download_from_drive.py.

Usage:
    python imagery/export_to_drive.py --project YOUR_GEE_PROJECT_ID

    # Monitor task progress:
    python imagery/export_to_drive.py --project YOUR_GEE_PROJECT_ID --status

    # After all tasks complete, download:
    python imagery/download_from_drive.py --project YOUR_GEE_PROJECT_ID
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parents[1]
PAIRS_CSV = ROOT / "outputs" / "split_pairs" / "split_pairs_k02_to_k30.csv"
OUT_IMAGES = ROOT / "outputs" / "images"
LOG_DIR   = ROOT / "logs"
TASK_JSON = ROOT / "logs" / "export_tasks.json"  # maps grid_id → task id
LOG_DIR.mkdir(parents=True, exist_ok=True)

# SHP_cleaned geometry, produced by ../../../../02_morphology_metrics/.
# Override with AFRICA_SHP_CLEANED_ROOT if your copy lives elsewhere.
FULL_GRID_ROOT = Path(os.environ.get(
    "AFRICA_SHP_CLEANED_ROOT",
    "../../../../02_morphology_metrics/outputs/SHP_cleaned",
))

# GEE config — must match generate_images.py
START_DATE      = "2024-01-01"
END_DATE        = "2024-06-30"
MAX_CLOUD_PCT   = 30.0
SCALE_M         = 10
SATELLITE_BUFFER_M = 20.0
S2_COLLECTION   = "COPERNICUS/S2_SR_HARMONIZED"
SCL_MASK_VALUES = [3, 8, 9, 10]

# Google Drive folder to export into
DRIVE_FOLDER = "GEE_K_Selection_Exports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "export_to_drive.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def _fs_name(name: str) -> str:
    return name.replace("/", "_")


_shp_cache: dict = {}

def get_cell_geometry(grid_id: str, city: str, country: str):
    key = (country, city)
    if key not in _shp_cache:
        city_fs  = _fs_name(city)
        shp_dir  = FULL_GRID_ROOT / country / city_fs
        shp_files = list(shp_dir.glob("*.shp")) if shp_dir.exists() else []
        _shp_cache[key] = gpd.read_file(shp_files[0]) if shp_files else None
    gdf = _shp_cache[key]
    if gdf is None:
        return None, None
    row = gdf[gdf["GridID"] == grid_id]
    if row.empty:
        return None, None
    geom = row.geometry.iloc[0]
    crs  = gdf.crs
    return geom, crs


def make_composite(geometry_ee: ee.Geometry) -> ee.Image:
    def mask_clouds(img):
        scl  = img.select("SCL")
        mask = scl.neq(SCL_MASK_VALUES[0])
        for v in SCL_MASK_VALUES[1:]:
            mask = mask.And(scl.neq(v))
        return img.updateMask(mask)

    col = (ee.ImageCollection(S2_COLLECTION)
           .filterBounds(geometry_ee)
           .filterDate(START_DATE, END_DATE)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT))
           .map(mask_clouds)
           .select(["B4", "B3", "B2"]))

    composite = col.median()
    # Scale reflectance → uint8
    composite = (composite.multiply(0.0001)
                          .clamp(0, 0.3)
                          .multiply(255 / 0.3)
                          .uint8())
    return composite.clip(geometry_ee)


def submit_export(grid_id: str, city: str, country: str) -> str | None:
    """Submit one Export.image.toDrive task. Returns task id or None on error."""
    geom, crs = get_cell_geometry(grid_id, city, country)
    if geom is None:
        log.warning(f"Geometry not found: {country}/{city}/{grid_id}")
        return None

    # Reproject to WGS84 for GEE
    if crs.to_epsg() != 4326:
        gdf_wgs = gpd.GeoDataFrame({"geometry": [geom]}, crs=crs).to_crs(4326)
        geom_wgs = gdf_wgs.geometry.iloc[0]
    else:
        geom_wgs = geom

    # Add buffer in degrees (~20m at equator ≈ 0.00018°)
    geom_buffered = geom_wgs.buffer(SATELLITE_BUFFER_M / 111_320)
    coords = list(geom_buffered.exterior.coords)
    ee_geom = ee.Geometry.Polygon([list(c) for c in coords])

    try:
        composite = make_composite(ee_geom)
        task = ee.batch.Export.image.toDrive(
            image=composite,
            description=grid_id,
            folder=DRIVE_FOLDER,
            fileNamePrefix=grid_id,
            region=ee_geom,
            scale=SCALE_M,
            crs="EPSG:4326",
            fileFormat="GeoTIFF",
            maxPixels=1e8,
        )
        task.start()
        return task.id
    except Exception as e:
        log.warning(f"Submit failed {grid_id}: {e}")
        return None


def load_tasks() -> dict:
    if TASK_JSON.exists():
        return json.loads(TASK_JSON.read_text())
    return {}


def save_tasks(tasks: dict):
    TASK_JSON.write_text(json.dumps(tasks, indent=2))


def cmd_submit(args):
    ee.Initialize(project=args.project)
    log.info(f"Earth Engine initialized (project={args.project})")

    df = pd.read_csv(PAIRS_CSV)
    cells_a = df[["grid_id_a", "city_a", "country_a"]].rename(
        columns={"grid_id_a": "grid_id", "city_a": "city", "country_a": "country"})
    cells_b = df[["grid_id_b", "city_b", "country_b"]].rename(
        columns={"grid_id_b": "grid_id", "city_b": "city", "country_b": "country"})
    cells = (pd.concat([cells_a, cells_b])
               .drop_duplicates("grid_id")
               .reset_index(drop=True))

    # Skip already-rendered images and already-submitted tasks
    existing_tasks = load_tasks()
    skip_rendered  = {p.stem for p in OUT_IMAGES.rglob("*.png")}

    todo = [r for r in cells.itertuples()
            if r.grid_id not in existing_tasks and r.grid_id not in skip_rendered]
    log.info(f"Total cells: {len(cells)}  |  already submitted: {len(existing_tasks)}  "
             f"|  already rendered: {len(skip_rendered)}  |  to submit: {len(todo)}")

    submitted, errors = 0, 0
    for i, r in enumerate(todo):
        task_id = submit_export(r.grid_id, r.city, r.country)
        if task_id:
            existing_tasks[r.grid_id] = {
                "task_id": task_id,
                "city": r.city,
                "country": r.country,
                "status": "SUBMITTED",
            }
            submitted += 1
            if submitted % 50 == 0:
                save_tasks(existing_tasks)
                log.info(f"  Submitted {submitted}/{len(todo)} ...")
        else:
            errors += 1
        # Avoid GEE rate-limit on task submission (~1 task/s is safe)
        if i % 10 == 9:
            time.sleep(2)

    save_tasks(existing_tasks)
    log.info(f"Done. Submitted={submitted}, errors={errors}")
    log.info(f"Task list saved → {TASK_JSON}")
    log.info(f"Monitor with: python export_to_drive.py --project {args.project} --status")


def cmd_status(args):
    ee.Initialize(project=args.project)
    tasks = load_tasks()
    if not tasks:
        log.info("No tasks found. Run without --status to submit.")
        return

    # Build lookup from task_id → grid_id
    id_to_gid = {info["task_id"]: gid for gid, info in tasks.items()}

    log.info(f"Fetching task list from GEE ({len(tasks)} tasks submitted) …")
    gee_tasks = ee.batch.Task.list()
    log.info(f"  GEE returned {len(gee_tasks)} tasks")

    from collections import Counter
    counts = Counter()
    for t in gee_tasks:
        st = t.status()
        tid = st.get("id", "")
        state = st.get("state", "UNKNOWN")
        gid = id_to_gid.get(tid)
        if gid:
            tasks[gid]["status"] = state
            counts[state] += 1

    save_tasks(tasks)
    total = len(tasks)
    log.info(f"Task status ({total} total):")
    for state, n in sorted(counts.items()):
        log.info(f"  {state}: {n}")

    completed = counts.get("COMPLETED", 0)
    log.info(f"  Completion: {completed}/{total} ({completed/total*100:.1f}%)")
    failed_ids = [gid for gid, info in tasks.items() if info.get("status") == "FAILED"]
    if failed_ids:
        log.info(f"  Failed: {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--status", action="store_true",
                        help="Check status of submitted tasks instead of submitting")
    parser.add_argument("--pairs-csv", type=Path, default=None,
                        help="Override PAIRS_CSV path")
    parser.add_argument("--task-json", type=Path, default=None,
                        help="Override TASK_JSON path")
    parser.add_argument("--out-images", type=Path, default=None,
                        help="Override OUT_IMAGES path")
    parser.add_argument("--drive-folder", type=str, default=None,
                        help="Override Google Drive folder name")
    args = parser.parse_args()

    if args.pairs_csv:   PAIRS_CSV    = args.pairs_csv
    if args.task_json:   TASK_JSON    = args.task_json
    if args.out_images:  OUT_IMAGES   = args.out_images
    if args.drive_folder: DRIVE_FOLDER = args.drive_folder

    if args.status:
        cmd_status(args)
    else:
        cmd_submit(args)
