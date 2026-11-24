"""
generate_images.py
------------------
For each grid cell listed in split_pairs_k05_to_k30.csv, generate a composite
image identical to the png_integrated style (satellite + black building/road
overlay + hexagon mask) used for expert annotation throughout this project.

Rendering reuses the public render_integrated()/apply_hexagon_mask() and
GridCell/GridCellImageDownloader classes from ../../../../03_morphology_imagery/,
so output images are visually identical to the teacher-model training
set produced by that package.

Output: outputs/images/{city}/{grid_id}.png  (1000×1000 RGBA PNG)

Usage:
    python imagery/generate_images.py --project YOUR_GEE_PROJECT_ID
    python imagery/generate_images.py --project YOUR_GEE_PROJECT_ID --workers 4
"""

import argparse
import logging
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[1]
PAIRS_CSV  = ROOT / "outputs" / "split_pairs" / "split_pairs_k02_to_k30.csv"
OUT_IMAGES = ROOT / "outputs" / "images"
LOG_DIR    = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_IMAGES.mkdir(parents=True, exist_ok=True)

# ../../../../03_morphology_imagery/. Override with
# AFRICA_MORPHOLOGY_IMAGERY_ROOT if your copy lives elsewhere.
AFRICA_MORPHOLOGY_IMAGERY = Path(os.environ.get(
    "AFRICA_MORPHOLOGY_IMAGERY_ROOT",
    "../../../../03_morphology_imagery",
))

# SHP_cleaned geometry and the per-city building/road-network intermediates,
# both produced by ../../../../02_morphology_metrics/. Override with
# AFRICA_SHP_CLEANED_ROOT / AFRICA_MORPHOLOGY_METRICS_DATA_ROOT if your
# copy lives elsewhere.
FULL_GRID_ROOT    = Path(os.environ.get(
    "AFRICA_SHP_CLEANED_ROOT",
    "../../../../02_morphology_metrics/outputs/SHP_cleaned",
))
MOMEPY_DATA_ROOT  = Path(os.environ.get(
    "AFRICA_MORPHOLOGY_METRICS_DATA_ROOT",
    "../../../../02_morphology_metrics/data",
))

sys.path.insert(0, str(AFRICA_MORPHOLOGY_IMAGERY))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "generate_images.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from ../../../../03_morphology_imagery/
# ---------------------------------------------------------------------------
try:
    import geopandas as gpd
    from PIL import Image
    import ee
    from satellite_download import GridCell, GridCellImageDownloader
    from render import render_integrated, clip_to_cell as _amri_clip_to_cell
    HAS_DEPS = True
except ImportError as e:
    log.error(f"Missing dependency: {e}")
    HAS_DEPS = False

SATELLITE_BUFFER_M   = 20.0

# ---------------------------------------------------------------------------
# Vector layer paths — derived from Momepy_workflow/Metrix/data directory
# ---------------------------------------------------------------------------

def _fs_name(name: str) -> str:
    """Convert city/country name to filesystem-safe form (replace '/' with '_')."""
    return name.replace("/", "_")


def get_vector_paths(city: str, country: str) -> tuple[str, str]:
    """Return (buildings_shp_path, highways_geojson_path) for a city."""
    city_fs   = _fs_name(city)
    city_dir  = MOMEPY_DATA_ROOT / country / city_fs
    buildings = city_dir / "Output" / "open_buildings.shp"
    highways  = city_dir / "highways" / f"{city_fs}_highways.geojson"
    return str(buildings), str(highways)

# ---------------------------------------------------------------------------
# Geometry lookup from full-Africa SHP_cleaned
# ---------------------------------------------------------------------------
_shp_cache: dict = {}

def get_cell_row(grid_id: str, city: str, country: str):
    """Return (geometry, crs_str, center_x, center_y) for a grid cell."""
    key = (country, city)
    if key not in _shp_cache:
        city_fs = _fs_name(city)
        shp_dir = FULL_GRID_ROOT / country / city_fs
        shp_files = list(shp_dir.glob("*.shp")) if shp_dir.exists() else []
        if not shp_files:
            log.warning(f"No SHP for {country}/{city}")
            _shp_cache[key] = None
        else:
            _shp_cache[key] = gpd.read_file(shp_files[0])
    gdf = _shp_cache[key]
    if gdf is None:
        return None
    row = gdf[gdf["GridID"] == grid_id]
    if row.empty:
        return None
    geom = row.geometry.iloc[0]
    crs  = gdf.crs.to_string()
    cx   = geom.centroid.x
    cy   = geom.centroid.y
    return geom, crs, cx, cy

# ---------------------------------------------------------------------------
# Core rendering — clip_to_cell and the render_integrated/apply_hexagon_mask
# pair used below both come from ../../../../03_morphology_imagery/ (see
# imports above), so this script only wires them up per cell.
# ---------------------------------------------------------------------------
clip_to_cell = _amri_clip_to_cell

DOWNLOAD_TIMEOUT_S = 60  # seconds before treating a download as hung

def download_satellite_for_cell(grid_cell, downloader, max_retries=3):
    import time
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    gdf = gpd.GeoDataFrame({"geometry": [grid_cell.geometry]}, crs=grid_cell.crs)
    buffered_bounds = gdf.geometry.iloc[0].buffer(SATELLITE_BUFFER_M).bounds

    for attempt in range(1, max_retries + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as _pool:
                fut = _pool.submit(downloader.download_grid_cell, grid_cell, buffer_m=SATELLITE_BUFFER_M)
                result = fut.result(timeout=DOWNLOAD_TIMEOUT_S)
            if result.success and result.output_path and result.output_path.exists():
                sat_img = Image.open(result.output_path).convert("RGB")
                return sat_img, buffered_bounds
            err = getattr(result, "error", "unknown")
            log.warning(f"Satellite download failed (attempt {attempt}/{max_retries}) "
                        f"for {grid_cell.grid_id}: {err}")
        except FuturesTimeout:
            log.warning(f"Satellite timeout ({DOWNLOAD_TIMEOUT_S}s, attempt {attempt}/{max_retries}) "
                        f"for {grid_cell.grid_id}")
        except Exception as e:
            log.warning(f"Satellite error (attempt {attempt}/{max_retries}) "
                        f"for {grid_cell.grid_id}: {e}")
        if attempt < max_retries:
            time.sleep(3)

    return None, None

# ---------------------------------------------------------------------------
# Per-cell worker
# ---------------------------------------------------------------------------
_buildings_cache: dict = {}
_highways_cache:  dict = {}

def get_vector_layers(city: str, country: str, target_crs: str):
    """Load and cache buildings + highways for a city (keyed by country/city)."""
    cache_key = (country, city)
    if cache_key not in _buildings_cache:
        bldg_path, road_path = get_vector_paths(city, country)

        if Path(bldg_path).exists():
            bldg = gpd.read_file(bldg_path)
            if bldg.crs.to_string() != target_crs:
                bldg = bldg.to_crs(target_crs)
            _buildings_cache[cache_key] = bldg
        else:
            log.warning(f"Buildings not found for {country}/{city}: {bldg_path}")
            _buildings_cache[cache_key] = gpd.GeoDataFrame(geometry=[])

        if Path(road_path).exists():
            road = gpd.read_file(road_path)
            if road.crs.to_string() != target_crs:
                road = road.to_crs(target_crs)
            _highways_cache[cache_key] = road
        else:
            log.warning(f"Highways not found for {country}/{city}: {road_path}")
            _highways_cache[cache_key] = gpd.GeoDataFrame(geometry=[])

    return _buildings_cache[cache_key], _highways_cache[cache_key]


def download_cell(grid_id: str, city: str, country: str,
                  downloader) -> dict:
    """Download satellite + load vector data. Returns a payload dict for rendering.
    Safe to call from multiple threads (no matplotlib)."""
    out_path = OUT_IMAGES / _fs_name(country) / _fs_name(city) / f"{grid_id}.png"
    if out_path.exists():
        return {"grid_id": grid_id, "cached": True}

    result = get_cell_row(grid_id, city, country)
    if result is None:
        return {"grid_id": grid_id, "error": "geometry not found"}
    cell_geom, crs, cx, cy = result

    buildings, highways = get_vector_layers(city, country, crs)
    buildings_clipped = clip_to_cell(buildings, cell_geom)
    highways_clipped  = clip_to_cell(highways,  cell_geom)

    grid_cell_obj = GridCell(
        grid_id=grid_id, city=city, country=country,
        crs=crs, center_x=cx, center_y=cy, geometry=cell_geom
    )
    sat_img, sat_bounds = download_satellite_for_cell(grid_cell_obj, downloader)

    return {
        "grid_id": grid_id,
        "out_path": out_path,
        "cell_geom": cell_geom,
        "crs": crs,
        "buildings_clipped": buildings_clipped,
        "highways_clipped": highways_clipped,
        "sat_img": sat_img,
        "sat_bounds": sat_bounds,
    }


def render_cell(payload: dict) -> tuple[str, bool, str]:
    """Render one cell from a pre-fetched payload. Must run in a single thread."""
    grid_id = payload["grid_id"]
    if payload.get("cached"):
        return grid_id, True, "cached"
    if "error" in payload:
        return grid_id, False, payload["error"]
    try:
        payload["out_path"].parent.mkdir(parents=True, exist_ok=True)
        render_integrated(
            payload["cell_geom"],
            payload["buildings_clipped"],
            payload["highways_clipped"],
            payload["sat_img"],
            payload["sat_bounds"],
            payload["out_path"],
            payload["crs"],
        )
        return grid_id, True, "ok"
    except Exception as e:
        return grid_id, False, str(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="GEE project ID")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel satellite download workers (rendering is single-threaded)")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Exit after generating this many new images (0 = no limit, for auto-restart loops)")
    parser.add_argument("--pairs-csv", type=Path, default=None,
                        help="Override PAIRS_CSV path")
    parser.add_argument("--out-images", type=Path, default=None,
                        help="Override OUT_IMAGES path")
    args = parser.parse_args()

    global PAIRS_CSV, OUT_IMAGES
    if args.pairs_csv:
        PAIRS_CSV = args.pairs_csv
    if args.out_images:
        OUT_IMAGES = args.out_images
        OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    if not HAS_DEPS:
        sys.exit(1)

    try:
        ee.Initialize(project=args.project)
        log.info(f"Earth Engine initialised (project={args.project})")
    except Exception as e:
        log.error(f"Earth Engine init failed: {e}")
        sys.exit(1)

    if not PAIRS_CSV.exists():
        log.error(f"Pairs CSV not found: {PAIRS_CSV}")
        sys.exit(1)

    df = pd.read_csv(PAIRS_CSV)
    cells_a = df[["grid_id_a", "city_a", "country_a"]].rename(
        columns={"grid_id_a": "grid_id", "city_a": "city", "country_a": "country"})
    cells_b = df[["grid_id_b", "city_b", "country_b"]].rename(
        columns={"grid_id_b": "grid_id", "city_b": "city", "country_b": "country"})
    cells = (pd.concat([cells_a, cells_b])
               .drop_duplicates("grid_id")
               .reset_index(drop=True))
    log.info(f"Unique grid cells to process: {len(cells)}")

    downloader = GridCellImageDownloader(
        output_dir=OUT_IMAGES / "_satellite_cache",
        start_date="2024-01-01",
        end_date="2024-06-30",
        max_cloud_percent=30.0,
        num_workers=1,
        logger=log,
    )
    downloader.setup_earth_engine()

    tasks = [(r.grid_id, r.city, r.country, downloader)
             for r in cells.itertuples()]
    batch_size = args.batch_size

    # Phase 1: parallel satellite downloads (thread-safe, no matplotlib)
    # Phase 2: serial rendering (matplotlib is NOT thread-safe)
    # We pipeline these: download N ahead, render as results come in.
    import queue as _queue
    render_queue = _queue.Queue(maxsize=args.workers * 2)
    success, failed, new_count = 0, 0, 0
    total = len(tasks)

    def producer():
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download_cell, *t): t[0] for t in tasks}
            for fut in as_completed(futures):
                render_queue.put(fut.result())
        render_queue.put(None)  # sentinel

    import threading as _threading
    prod_thread = _threading.Thread(target=producer, daemon=True)
    prod_thread.start()

    done = 0
    while True:
        payload = render_queue.get()
        if payload is None:
            break
        gid, ok, msg = render_cell(payload)
        done += 1
        if ok:
            success += 1
            if msg != "cached":
                new_count += 1
                if new_count % 50 == 0:
                    log.info(f"Progress: {done}/{total} done (new={new_count})")
        else:
            failed += 1
            log.warning(f"FAILED {gid}: {msg}")

    prod_thread.join()
    log.info(f"\nDone. success={success}, failed={failed}, total={total}")
    log.info(f"Images saved to: {OUT_IMAGES}")


if __name__ == "__main__":
    main()
