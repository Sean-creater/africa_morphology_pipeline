"""Pre-clip and reproject each raw global/regional raster source to per-country
extents, so that per-city grid generation can read a small country-level file
instead of masking the full-extent source raster for every city.

This is a preprocessing step that runs once per country (results are cached
to disk and skipped on subsequent runs) and is a prerequisite for
`pipeline.GridPipeline`, which expects to find its raster inputs under
`{country_tif_dir}/{Country_With_Underscores}/{output_name}.tif` (see
config.COUNTRY_TIF_MAPPING for the exact filenames expected).

Usage:
    python -m africa_hex_grid_workflow.clip_country_rasters \\
        --country-boundaries /path/to/country_boundaries.gpkg \\
        --raw-data-dir /path/to/raw_rasters \\
        --output-dir /path/to/00_country_TIF \\
        --country-name-column NAME_0
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import geopandas as gpd
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import Resampling, calculate_default_transform, reproject
from tqdm import tqdm

logger = logging.getLogger(__name__)

#: Target CRS for the country-level raster cache. This is intentionally a
#: single fixed continental projection (distinct from the per-city dynamic
#: UTM zones used for the actual grid) because the cache is a shared,
#: country-scale artifact reused across every city within that country;
#: `pipeline.GridPipeline` reprojects the grid to match each raster's CRS
#: at attachment time regardless of what this value is.
TARGET_CRS = "ESRI:102022"

#: Source raster patterns, relative to the raw-data root, one subdirectory
#: per data source (matching config.DATA_SOURCES' `source_dir` values), and
#: the output filename each produces (matching config.COUNTRY_TIF_MAPPING).
SOURCES: Dict[str, Dict[str, str]] = {
    "building_height": {
        "source_dir": "WSF3D",
        "source_pattern": "WSF3D_V02_BuildingHeight.tif",
        "output_name": "building_height.tif",
    },
    "building_volume": {
        "source_dir": "WSF3D",
        "source_pattern": "WSF3D_V02_BuildingVolume.tif",
        "output_name": "building_volume.tif",
    },
    "population_2022": {
        "source_dir": "WorldPop",
        "source_pattern": "sub_saharan_africa_ppp_100m_2022.tif",
        "output_name": "population_2022.tif",
    },
    "land_cover_discrete": {
        "source_dir": "Land Cover 2019 (raster 100 m), global, yearly – version 3",
        "source_pattern": "PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif",
        "output_name": "land_cover_discrete.tif",
    },
    "lc_fraction_bare": {
        "source_dir": "Land Cover 2019 (raster 100 m), global, yearly – version 3",
        "source_pattern": "PROBAV_LC100_global_v3.0.1_2019-nrt_Bare-CoverFraction-layer_EPSG-4326.tif",
        "output_name": "lc_bare.tif",
    },
    "lc_fraction_builtup": {
        "source_dir": "Land Cover 2019 (raster 100 m), global, yearly – version 3",
        "source_pattern": "PROBAV_LC100_global_v3.0.1_2019-nrt_BuiltUp-CoverFraction-layer_EPSG-4326.tif",
        "output_name": "lc_builtup.tif",
    },
    "lc_fraction_crops": {
        "source_dir": "Land Cover 2019 (raster 100 m), global, yearly – version 3",
        "source_pattern": "PROBAV_LC100_global_v3.0.1_2019-nrt_Crops-CoverFraction-layer_EPSG-4326.tif",
        "output_name": "lc_crops.tif",
    },
    "lc_fraction_grass": {
        "source_dir": "Land Cover 2019 (raster 100 m), global, yearly – version 3",
        "source_pattern": "PROBAV_LC100_global_v3.0.1_2019-nrt_Grass-CoverFraction-layer_EPSG-4326.tif",
        "output_name": "lc_grass.tif",
    },
    "lc_fraction_shrub": {
        "source_dir": "Land Cover 2019 (raster 100 m), global, yearly – version 3",
        "source_pattern": "PROBAV_LC100_global_v3.0.1_2019-nrt_Shrub-CoverFraction-layer_EPSG-4326.tif",
        "output_name": "lc_shrub.tif",
    },
    "lc_fraction_tree": {
        "source_dir": "Land Cover 2019 (raster 100 m), global, yearly – version 3",
        "source_pattern": "PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif",
        "output_name": "lc_tree.tif",
    },
    "lc_fraction_water": {
        "source_dir": "Land Cover 2019 (raster 100 m), global, yearly – version 3",
        "source_pattern": "PROBAV_LC100_global_v3.0.1_2019-nrt_PermanentWater-CoverFraction-layer_EPSG-4326.tif",
        "output_name": "lc_water.tif",
    },
    "lc_fraction_mosslichen": {
        "source_dir": "Land Cover 2019 (raster 100 m), global, yearly – version 3",
        "source_pattern": "PROBAV_LC100_global_v3.0.1_2019-nrt_MossLichen-CoverFraction-layer_EPSG-4326.tif",
        "output_name": "lc_mosslichen.tif",
    },
    "lc_fraction_seasonwater": {
        "source_dir": "Land Cover 2019 (raster 100 m), global, yearly – version 3",
        "source_pattern": "PROBAV_LC100_global_v3.0.1_2019-nrt_SeasonalWater-CoverFraction-layer_EPSG-4326.tif",
        "output_name": "lc_seasonwater.tif",
    },
    "lc_fraction_snow": {
        "source_dir": "Land Cover 2019 (raster 100 m), global, yearly – version 3",
        "source_pattern": "PROBAV_LC100_global_v3.0.1_2019-nrt_Snow-CoverFraction-layer_EPSG-4326.tif",
        "output_name": "lc_snow.tif",
    },
}


@dataclass
class ClipResult:
    source_key: str
    succeeded: list
    failed: list
    skipped: list


def _safe_country_dirname(country_name: str) -> str:
    return country_name.replace(" ", "_").replace("'", "").replace("(", "").replace(")", "")


def clip_and_reproject(raster_path: Path, geometry, output_path: Path) -> bool:
    """Clip `raster_path` to `geometry` (assumed EPSG:4326) and reproject
    the clipped result to TARGET_CRS, writing to `output_path`.

    Clipping is done first, in the source raster's native CRS, then the
    (much smaller) clipped raster is reprojected — reprojecting the full
    global/regional raster before clipping would be far more expensive for
    no benefit, since only the clipped region is ever needed.
    """
    try:
        with rasterio.open(raster_path) as src:
            geom_gdf = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")
            if src.crs and geom_gdf.crs != src.crs:
                geom_gdf = geom_gdf.to_crs(src.crs)
            geom = geom_gdf.geometry.values[0]

            out_image, out_transform = rasterio_mask(
                src, [geom], crop=True, nodata=src.nodata if src.nodata is not None else 0, all_touched=True
            )
            out_meta = src.meta.copy()
            out_meta.update(
                {"height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform}
            )

            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                temp_path = Path(tmp.name)
            with rasterio.open(temp_path, "w", **out_meta) as temp_dst:
                temp_dst.write(out_image)

        try:
            with rasterio.open(temp_path) as temp_src:
                transform, width, height = calculate_default_transform(
                    temp_src.crs, TARGET_CRS, temp_src.width, temp_src.height, *temp_src.bounds
                )
                dst_meta = temp_src.meta.copy()
                dst_meta.update(
                    {
                        "crs": TARGET_CRS, "transform": transform, "width": width, "height": height,
                        "driver": "GTiff", "compress": "lzw",
                    }
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(output_path, "w", **dst_meta) as dst:
                    for band_idx in range(1, temp_src.count + 1):
                        reproject(
                            source=rasterio.band(temp_src, band_idx),
                            destination=rasterio.band(dst, band_idx),
                            src_transform=temp_src.transform,
                            src_crs=temp_src.crs,
                            dst_transform=transform,
                            dst_crs=TARGET_CRS,
                            resampling=Resampling.nearest,
                        )
        finally:
            temp_path.unlink(missing_ok=True)

        return True

    except Exception as exc:
        logger.error("Failed to clip/reproject %s: %s", raster_path.name, exc)
        return False


def clip_source_for_all_countries(
    source_key: str, source_config: Dict[str, str], countries_gdf: gpd.GeoDataFrame,
    raw_data_dir: Path, output_dir: Path, country_name_column: str,
) -> ClipResult:
    """Clip one raster source to every country in `countries_gdf`."""
    source_path = raw_data_dir / source_config["source_dir"] / source_config["source_pattern"]
    output_name = source_config["output_name"]
    result = ClipResult(source_key=source_key, succeeded=[], failed=[], skipped=[])

    if not source_path.exists():
        logger.warning("Source file not found, skipping all countries: %s", source_path)
        result.failed = countries_gdf[country_name_column].tolist()
        return result

    for _, row in tqdm(
        countries_gdf.iterrows(), total=len(countries_gdf), desc=f"Clipping {source_key}", unit="country"
    ):
        country_name = row[country_name_column]
        output_path = output_dir / _safe_country_dirname(country_name) / output_name

        if output_path.exists():
            result.skipped.append(country_name)
            continue

        if clip_and_reproject(source_path, row.geometry, output_path):
            result.succeeded.append(country_name)
        else:
            result.failed.append(country_name)

    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--country-boundaries", required=True, type=Path)
    parser.add_argument("--country-name-column", default="NAME_0")
    parser.add_argument("--raw-data-dir", required=True, type=Path, help="Root directory of raw source rasters")
    parser.add_argument("--output-dir", required=True, type=Path, help="Country-TIF cache output directory")
    parser.add_argument("--country", default=None, help="Process only this country (default: all)")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    countries_gdf = gpd.read_file(args.country_boundaries)
    if args.country:
        countries_gdf = countries_gdf[countries_gdf[args.country_name_column] == args.country]
        if len(countries_gdf) == 0:
            logger.error("Country '%s' not found in %s", args.country, args.country_boundaries)
            return 1

    for source_key, source_config in SOURCES.items():
        result = clip_source_for_all_countries(
            source_key, source_config, countries_gdf, args.raw_data_dir, args.output_dir, args.country_name_column
        )
        logger.info(
            "%s: %d succeeded, %d skipped, %d failed",
            source_key, len(result.succeeded), len(result.skipped), len(result.failed),
        )

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
