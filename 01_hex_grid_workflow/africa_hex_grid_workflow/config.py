"""Declarative configuration: data source definitions, land cover class tables,
resampling rules, and grid alignment constants.

Nothing in this module reads the filesystem or depends on any particular
machine's directory layout. Paths to boundary files, country-TIF caches, and
output directories are supplied by the caller (CLI arguments or a config
object in `pipeline.py`), not hardcoded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------

#: Hexagon side length in meters. This is the only supported grid cell size
#: dimension knob; the grid is always flat-top hexagonal (see hexgrid.py).
GRID_RESOLUTION_M = 250

#: Global grid origin, expressed in the Africa Albers Equal Area Conic
#: projection (ESRI:102022), in meters.
#:
#: Every city or country is gridded independently, each in its own optimal
#: UTM projection (see pipeline.py:detect_utm_crs). Without a shared
#: reference point, two independently-generated grids would not align at
#: their shared boundary even if projected to the same CRS afterwards,
#: because hexagon centers are only guaranteed to tile seamlessly relative to
#: a single fixed origin. Anchoring every grid's axial coordinate system to
#: the same origin (translated into whichever CRS is used for that run)
#: ensures that grids produced by separate, independent processing runs
#: still tile consistently if reprojected into a common CRS later.
GLOBAL_GRID_ORIGIN_X = -3_000_000.0  # meters; covers West Africa
GLOBAL_GRID_ORIGIN_Y = -2_000_000.0  # meters; covers Sub-Saharan Africa

#: GridID format for hexagons produced with country/city context:
#: "{ISO3}_{city_index:04d}_{cell_number:06d}", e.g. "GHA_0005_000123".
GRID_ID_FORMAT = "{iso3}_{city_index:04d}_{cell_number:06d}"

# ---------------------------------------------------------------------------
# Numerical precision / dtype optimization
# ---------------------------------------------------------------------------

#: Decimal places to round each data type to before writing output. Applied
#: before dtype conversion so rounding error is bounded by the target dtype.
PRECISION_CONFIG: Dict[str, int] = {
    "coordinates": 2,
    "building_height": 2,
    "building_volume": 2,
    "population": 2,
    "land_cover_fraction": 2,
    "default": 2,
}

#: Storage dtype for each data type. float32 is sufficient precision for all
#: quantities here (heights in meters, percentages, population counts) and
#: halves file size relative to float64.
DTYPE_CONFIG: Dict[str, str] = {
    "coordinates": "float32",
    "building_height": "float32",
    "building_volume": "float32",
    "population": "float32",
    "land_cover_fraction": "float32",
    "default": "float32",
}

# ---------------------------------------------------------------------------
# Land cover class tables (Copernicus Global Land Cover 100m v3, 2019 epoch)
# ---------------------------------------------------------------------------

#: Class code -> descriptive name, per the Copernicus Global Land Cover 100m
#: v3 discrete classification legend.
LAND_COVER_CLASSES: Dict[int, str] = {
    20: "Shrubs", 30: "Herbaceous", 40: "Cropland", 50: "BuiltUp",
    60: "Bare", 70: "Snow_Ice", 80: "Water_Permanent", 90: "Wetland",
    100: "Moss_Lichen",
    111: "Forest_Closed_Evergreen_Needle", 112: "Forest_Closed_Evergreen_Broad",
    113: "Forest_Closed_Deciduous_Needle", 114: "Forest_Closed_Deciduous_Broad",
    115: "Forest_Closed_Mixed", 116: "Forest_Closed_Undefined",
    121: "Forest_Open_Evergreen_Needle", 122: "Forest_Open_Evergreen_Broad",
    123: "Forest_Open_Deciduous_Needle", 124: "Forest_Open_Deciduous_Broad",
    125: "Forest_Open_Mixed", 126: "Forest_Open_Undefined",
    200: "Ocean_Sea",
}

#: Class code -> shapefile-safe column name (Shapefile DBF fields are limited
#: to 10 characters), used for the LC_* percentage columns.
LAND_COVER_COLUMN_NAMES: Dict[int, str] = {
    20: "LC_Shrub", 30: "LC_Herb", 40: "LC_Crop", 50: "LC_Built",
    60: "LC_Bare", 70: "LC_Snow", 80: "LC_Water", 90: "LC_Wetlnd",
    100: "LC_Moss",
    111: "LC_FCEvNd", 112: "LC_FCEvBr", 113: "LC_FCDcNd", 114: "LC_FCDcBr",
    115: "LC_FCMix", 116: "LC_FCUnd",
    121: "LC_FOEvNd", 122: "LC_FOEvBr", 123: "LC_FODcNd", 124: "LC_FODcBr",
    125: "LC_FOMix", 126: "LC_FOUnd",
    200: "LC_Ocean",
}

# ---------------------------------------------------------------------------
# Data source definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSource:
    """One raster data source and how to extract it into a grid column.

    Attributes:
        source_pattern: Filename of the source raster (matched under the
            country-TIF cache directory for the relevant `source_dir`).
        output_column: Output column name (or column-name prefix, for
            multi-column sources) written to the grid.
        resampling: Aggregation method applied to valid pixels within each
            hexagon: "max", "min", "mean", "median", "sum", "std",
            "categorical_percentage" (land cover only), or
            "area_weighted_sum" (population only — apportions each pixel's
            value across intersecting hexagons in proportion to its
            covered area, rather than counting whole pixels that merely
            touch the hexagon; see raster_extract.py).
        description: Human-readable description, used in logs.
        nodata_values: Explicit sentinel values to treat as NoData, in
            addition to the raster's own `nodata` tag, NaN, and inf.
        scale_factor: Multiplier applied to valid pixel values before
            aggregation. WSF3D building height/volume are stored as int16
            with an implicit 0.1 scale factor (DLR's native encoding) — this
            is NOT a workaround, it is how the source data is distributed.
        valid_range: (min, max) inclusive bounds; pixels outside this range
            are treated as NoData. Distinguishes "legitimately valid data"
            from encoding artifacts distinct from the explicit nodata tag.
        filter_negative: If True, any negative pixel value is treated as
            NoData. Needed for WSF3D height/volume, where negative values
            that survive the nodata/valid_range filters still indicate
            sensor/processing artifacts rather than real measurements.
        source_dir: Subdirectory (under the raw-data root) holding the
            source raster, used only by the country-clipping step.
        data_type: Key into PRECISION_CONFIG / DTYPE_CONFIG.
        output_type: "single_column" (default) or "multi_column" for land
            cover, which expands into one percentage column per class.
    """

    source_pattern: str
    output_column: str
    resampling: str
    description: str
    nodata_values: Tuple[float, ...] = ()
    scale_factor: float = 1.0
    valid_range: Optional[Tuple[float, float]] = None
    filter_negative: bool = False
    source_dir: str = ""
    data_type: str = "default"
    output_type: str = "single_column"


#: All raster sources attached to the grid, keyed by an internal source key.
#: Preserve these values exactly: nodata values, scale factors, and valid
#: ranges encode real facts about how each source dataset is distributed,
#: not arbitrary tuning choices.
DATA_SOURCES: Dict[str, DataSource] = {
    # --- Building height (WSF3D), four aggregations of the same raster ---
    "building_height_max": DataSource(
        source_pattern="WSF3D_V02_BuildingHeight.tif",
        output_column="H_max",
        resampling="max",
        description="Building height (max) from WSF3D",
        nodata_values=(-32767.0, 0),
        scale_factor=0.1,
        valid_range=(0.1, 1000),
        filter_negative=True,
        source_dir="WSF3D",
        data_type="building_height",
    ),
    "building_height_mean": DataSource(
        source_pattern="WSF3D_V02_BuildingHeight.tif",
        output_column="H_mean",
        resampling="mean",
        description="Building height (mean) from WSF3D",
        nodata_values=(-32767.0, 0),
        scale_factor=0.1,
        valid_range=(0.1, 1000),
        filter_negative=True,
        source_dir="WSF3D",
        data_type="building_height",
    ),
    "building_height_std": DataSource(
        source_pattern="WSF3D_V02_BuildingHeight.tif",
        output_column="H_std",
        resampling="std",
        description="Building height (std) from WSF3D",
        nodata_values=(-32767.0, 0),
        scale_factor=0.1,
        valid_range=(0.1, 1000),
        filter_negative=True,
        source_dir="WSF3D",
        data_type="building_height",
    ),
    "building_height_sum": DataSource(
        source_pattern="WSF3D_V02_BuildingHeight.tif",
        output_column="H_sum",
        resampling="sum",
        description="Building height (sum, proxy for building density) from WSF3D",
        nodata_values=(-32767.0, 0),
        scale_factor=0.1,
        valid_range=(0.1, 1000),
        filter_negative=True,
        source_dir="WSF3D",
        data_type="building_height",
    ),
    # --- Building volume (WSF3D), four aggregations of the same raster ---
    "building_volume_sum": DataSource(
        source_pattern="WSF3D_V02_BuildingVolume.tif",
        output_column="V_sum",
        resampling="sum",
        description="Building volume (sum) from WSF3D",
        nodata_values=(-32767.0, 0),
        scale_factor=1.0,
        valid_range=(0.1, 1e9),
        filter_negative=True,
        source_dir="WSF3D",
        data_type="building_volume",
    ),
    "building_volume_mean": DataSource(
        source_pattern="WSF3D_V02_BuildingVolume.tif",
        output_column="V_mean",
        resampling="mean",
        description="Building volume (mean) from WSF3D",
        nodata_values=(-32767.0, 0),
        scale_factor=1.0,
        valid_range=(0.1, 1e9),
        filter_negative=True,
        source_dir="WSF3D",
        data_type="building_volume",
    ),
    "building_volume_max": DataSource(
        source_pattern="WSF3D_V02_BuildingVolume.tif",
        output_column="V_max",
        resampling="max",
        description="Building volume (max) from WSF3D",
        nodata_values=(-32767.0, 0),
        scale_factor=1.0,
        valid_range=(0.1, 1e9),
        filter_negative=True,
        source_dir="WSF3D",
        data_type="building_volume",
    ),
    "building_volume_std": DataSource(
        source_pattern="WSF3D_V02_BuildingVolume.tif",
        output_column="V_std",
        resampling="std",
        description="Building volume (std) from WSF3D",
        nodata_values=(-32767.0, 0),
        scale_factor=1.0,
        valid_range=(0.1, 1e9),
        filter_negative=True,
        source_dir="WSF3D",
        data_type="building_volume",
    ),
    # --- Land cover discrete classification: multi-column percentages ---
    "land_cover": DataSource(
        source_pattern=(
            "PROBAV_LC100_global_v3.0.1_2019-nrt_"
            "Discrete-Classification-map_EPSG-4326.tif"
        ),
        output_column="LC",
        resampling="categorical_percentage",
        description="Land cover discrete classification (Copernicus 100m v3)",
        nodata_values=(0, 255),  # 0 = Unknown, 255 = NoData
        scale_factor=1.0,
        valid_range=(1, 200),
        source_dir="Land Cover 2019 (raster 100 m), global, yearly – version 3",
        output_type="multi_column",
    ),
    # --- Population (WorldPop 2022, Sub-Saharan Africa, constrained) ---
    "population_2022": DataSource(
        source_pattern="sub_saharan_africa_ppp_100m_2022.tif",
        output_column="Pop2022",
        resampling="area_weighted_sum",
        description="Population count from WorldPop 2022 (Sub-Saharan Africa)",
        nodata_values=(-99999,),
        scale_factor=1.0,
        valid_range=(0, 1e6),
        source_dir="WorldPop",
        data_type="population",
    ),
    # --- Land cover cover-fraction layers: continuous 0-100% per pixel ---
    "lc_fraction_bare": DataSource(
        source_pattern="PROBAV_LC100_global_v3.0.1_2019-nrt_Bare-CoverFraction-layer_EPSG-4326.tif",
        output_column="LCF_Bare",
        resampling="mean",
        description="Bare/sparse vegetation cover fraction (0-100%)",
        nodata_values=(255,),
        valid_range=(0, 100),
        source_dir="Land Cover 2019 (raster 100 m), global, yearly – version 3",
        data_type="land_cover_fraction",
    ),
    "lc_fraction_builtup": DataSource(
        source_pattern="PROBAV_LC100_global_v3.0.1_2019-nrt_BuiltUp-CoverFraction-layer_EPSG-4326.tif",
        output_column="LCF_Built",
        resampling="mean",
        description="Built-up cover fraction (0-100%)",
        nodata_values=(255,),
        valid_range=(0, 100),
        source_dir="Land Cover 2019 (raster 100 m), global, yearly – version 3",
        data_type="land_cover_fraction",
    ),
    "lc_fraction_crops": DataSource(
        source_pattern="PROBAV_LC100_global_v3.0.1_2019-nrt_Crops-CoverFraction-layer_EPSG-4326.tif",
        output_column="LCF_Crop",
        resampling="mean",
        description="Cropland cover fraction (0-100%)",
        nodata_values=(255,),
        valid_range=(0, 100),
        source_dir="Land Cover 2019 (raster 100 m), global, yearly – version 3",
        data_type="land_cover_fraction",
    ),
    "lc_fraction_grass": DataSource(
        source_pattern="PROBAV_LC100_global_v3.0.1_2019-nrt_Grass-CoverFraction-layer_EPSG-4326.tif",
        output_column="LCF_Grass",
        resampling="mean",
        description="Herbaceous vegetation cover fraction (0-100%)",
        nodata_values=(255,),
        valid_range=(0, 100),
        source_dir="Land Cover 2019 (raster 100 m), global, yearly – version 3",
        data_type="land_cover_fraction",
    ),
    "lc_fraction_mosslichen": DataSource(
        source_pattern="PROBAV_LC100_global_v3.0.1_2019-nrt_MossLichen-CoverFraction-layer_EPSG-4326.tif",
        output_column="LCF_Moss",
        resampling="mean",
        description="Moss and lichen cover fraction (0-100%)",
        nodata_values=(255,),
        valid_range=(0, 100),
        source_dir="Land Cover 2019 (raster 100 m), global, yearly – version 3",
        data_type="land_cover_fraction",
    ),
    "lc_fraction_permwater": DataSource(
        source_pattern="PROBAV_LC100_global_v3.0.1_2019-nrt_PermanentWater-CoverFraction-layer_EPSG-4326.tif",
        output_column="LCF_Water",
        resampling="mean",
        description="Permanent water bodies cover fraction (0-100%)",
        nodata_values=(255,),
        valid_range=(0, 100),
        source_dir="Land Cover 2019 (raster 100 m), global, yearly – version 3",
        data_type="land_cover_fraction",
    ),
    "lc_fraction_seasonwater": DataSource(
        source_pattern="PROBAV_LC100_global_v3.0.1_2019-nrt_SeasonalWater-CoverFraction-layer_EPSG-4326.tif",
        output_column="LCF_SeaWtr",
        resampling="mean",
        description="Seasonal water bodies cover fraction (0-100%)",
        nodata_values=(255,),
        valid_range=(0, 100),
        source_dir="Land Cover 2019 (raster 100 m), global, yearly – version 3",
        data_type="land_cover_fraction",
    ),
    "lc_fraction_shrub": DataSource(
        source_pattern="PROBAV_LC100_global_v3.0.1_2019-nrt_Shrub-CoverFraction-layer_EPSG-4326.tif",
        output_column="LCF_Shrub",
        resampling="mean",
        description="Shrubland cover fraction (0-100%)",
        nodata_values=(255,),
        valid_range=(0, 100),
        source_dir="Land Cover 2019 (raster 100 m), global, yearly – version 3",
        data_type="land_cover_fraction",
    ),
    "lc_fraction_snow": DataSource(
        source_pattern="PROBAV_LC100_global_v3.0.1_2019-nrt_Snow-CoverFraction-layer_EPSG-4326.tif",
        output_column="LCF_Snow",
        resampling="mean",
        description="Snow and ice cover fraction (0-100%)",
        nodata_values=(255,),
        valid_range=(0, 100),
        source_dir="Land Cover 2019 (raster 100 m), global, yearly – version 3",
        data_type="land_cover_fraction",
    ),
    "lc_fraction_tree": DataSource(
        source_pattern="PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif",
        output_column="LCF_Tree",
        resampling="mean",
        description="Tree cover fraction (0-100%)",
        nodata_values=(255,),
        valid_range=(0, 100),
        source_dir="Land Cover 2019 (raster 100 m), global, yearly – version 3",
        data_type="land_cover_fraction",
    ),
}

#: Maps the filename produced by `clip_country_rasters.py` to the
#: `DATA_SOURCES` keys it supplies. One clipped country raster can back
#: several output columns (e.g. building_height.tif backs max/mean/std/sum).
COUNTRY_TIF_MAPPING: Dict[str, Tuple[str, ...]] = {
    "building_height.tif": (
        "building_height_max", "building_height_mean",
        "building_height_std", "building_height_sum",
    ),
    "building_volume.tif": (
        "building_volume_sum", "building_volume_mean",
        "building_volume_max", "building_volume_std",
    ),
    "population_2022.tif": ("population_2022",),
    "land_cover_discrete.tif": ("land_cover",),
    "lc_bare.tif": ("lc_fraction_bare",),
    "lc_builtup.tif": ("lc_fraction_builtup",),
    "lc_crops.tif": ("lc_fraction_crops",),
    "lc_grass.tif": ("lc_fraction_grass",),
    "lc_shrub.tif": ("lc_fraction_shrub",),
    "lc_tree.tif": ("lc_fraction_tree",),
    "lc_water.tif": ("lc_fraction_permwater",),
    "lc_mosslichen.tif": ("lc_fraction_mosslichen",),
    "lc_seasonwater.tif": ("lc_fraction_seasonwater",),
    "lc_snow.tif": ("lc_fraction_snow",),
}

# ---------------------------------------------------------------------------
# Resampling rules per data source
# ---------------------------------------------------------------------------

#: method + all_touched policy for each source. `all_touched=True` means a
#: pixel is included if the hexagon touches it at all, not only if the pixel
#: center falls inside the hexagon; see raster_extract.py for the adaptive
#: override applied when pixels are large relative to the grid cell.
HEXAGON_RESAMPLING_CONFIG: Dict[str, Dict[str, object]] = {
    "building_height_max": {"method": "max", "all_touched": True},
    "building_height_mean": {"method": "mean", "all_touched": True},
    "building_height_std": {"method": "std", "all_touched": True},
    "building_height_sum": {"method": "sum", "all_touched": True},
    "building_volume_sum": {"method": "sum", "all_touched": True},
    "building_volume_mean": {"method": "mean", "all_touched": True},
    "building_volume_max": {"method": "max", "all_touched": True},
    "building_volume_std": {"method": "std", "all_touched": True},
    "population_2022": {"method": "area_weighted_sum", "all_touched": True},
    "land_cover": {"method": "categorical_percentage", "all_touched": True},
    "lc_fraction_bare": {"method": "mean", "all_touched": True},
    "lc_fraction_builtup": {"method": "mean", "all_touched": True},
    "lc_fraction_crops": {"method": "mean", "all_touched": True},
    "lc_fraction_grass": {"method": "mean", "all_touched": True},
    "lc_fraction_mosslichen": {"method": "mean", "all_touched": True},
    "lc_fraction_permwater": {"method": "mean", "all_touched": True},
    "lc_fraction_seasonwater": {"method": "mean", "all_touched": True},
    "lc_fraction_shrub": {"method": "mean", "all_touched": True},
    "lc_fraction_snow": {"method": "mean", "all_touched": True},
    "lc_fraction_tree": {"method": "mean", "all_touched": True},
    "default": {"method": "max", "all_touched": True},
}

# ---------------------------------------------------------------------------
# Directory layout (relative names; caller supplies the data root)
# ---------------------------------------------------------------------------

DIR_STRUCTURE: Dict[str, str] = {
    "boundaries": "00_boundaries",
    "aoi": "01_aoi",
    "grids": "03_country_grids",
}
