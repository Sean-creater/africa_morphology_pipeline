"""
Building morphology analysis pipeline orchestrator.

Extracts buildings from Google Open Buildings (GCS), downloads and clips
OSM highway networks, calculates building/street-network/connectivity/
spatial-distribution metrics, and aggregates everything onto the 250 m
hexagonal grid produced by the companion grid-generation pipeline
(``africa_hex_grid_workflow``). Output is one enriched grid (SHP + CSV +
GeoJSON) per city, later consumed by the ``cleaning/`` stage to produce the
final standardized dataset.

Processing modes:
    single_city       - process one named city
    city_batch        - process a fixed list of cities
    country_batch      - process every city in one country (by ISO3)
    phased_automatic   - process all of Africa in three phases (see
                          ``config.PHASE1_SAMPLE_CITIES`` /
                          ``config.PHASE2_COUNTRY_ISO3``), with checkpointing
                          and automatic exclusion of countries that show no
                          Open Buildings coverage.

Cities are processed in a fixed order with no auto-skip on failure by
default (``STRICT_MODE``): if a city fails, the run halts so a completed
run is guaranteed to cover every requested city rather than silently
skipping ones that errored. Re-running resumes from the checkpoint.

Usage:
    python -m africa_morphology_metrics.pipeline
"""

from pathlib import Path
import logging
import sys
import time
import warnings
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from multiprocessing import cpu_count

# Shapefile field names longer than 10 characters are silently truncated by
# the driver; every output column name here is already <=10 characters
# (see data/reference/col_name_mapping.csv), so this warning is expected
# noise rather than a sign of data loss.
warnings.filterwarnings('ignore', message='.*Column names longer than.*')

from .trackers import NoBuildingDataTracker, ExcludedCountriesTracker
from .gcs_buildings import OpenBuildingsGCSProcessorOptimized as OpenBuildingsGCSProcessor
from .config import (
    OpenBuildingsGCSConfig,
    PHASE1_SAMPLE_CITIES,
    PHASE2_COUNTRY_ISO3,
    PHASE2_EXCLUDE_CITIES,
    ALL_AFRICA_ISO3,
    SUB_SAHARAN_AFRICA_ISO3,
)
from .osm_network import process_osm_highways_for_cities
from .global_data_cache import get_agglomeration_data, get_boundaries_data, clear_cache
from .checkpoint import ProcessingCheckpoint, get_checkpoint_path
from .morphology_metrics import (
    calculate_building_metrics,
    aggregate_metrics_to_grid,
    generate_column_mapping_with_categories,
)
from .streetnetwork_metrics import calculate_street_network_metrics_with_graph
from .connectivity import calculate_connectivity_metrics, USING_IGRAPH
from .spatial_distribution_metrics import calculate_and_aggregate_spatial_metrics
from .notifications import EmailNotifier
from .projection import get_crs_for_city, get_crs_for_country


# ============================================================================
# CONFIGURATION
# ============================================================================

def sanitize_filename(name: str) -> str:
    """
    Sanitize a city/country name for use as a path component.

    Replaces or strips characters that are invalid or awkward in file
    paths (slashes, colons, quotes, wildcard characters) while leaving the
    rest of the name intact, since city names are used directly as
    directory names throughout the output tree.
    """
    name = name.replace('/', '_').replace('\\', '_')
    name = name.replace(':', '_')
    for char in ['?', '*', '|', '"', '<', '>']:
        name = name.replace(char, '')
    name = name.strip().strip('_')
    return name


class Config:
    """
    Paths and parameters for one pipeline run.

    File paths below point at the shared workspace layout this pipeline
    was developed against (companion grid-generation output, AFRICAPOLIS
    agglomeration boundaries, country boundaries). Point these at your own
    copies - see the top-level README's "Data sources" section for where
    each one comes from and how to obtain it.
    """

    # ==================== PROCESSING MODE ====================
    # One of: "single_city", "city_batch", "country_batch", "phased_automatic"
    PROCESSING_MODE = "phased_automatic"

    CITY_NAME = "Kaolack"  # single_city mode
    TARGET_COUNTRY_ISO3 = "SEN"  # country_batch mode
    CITY_LIST = ["Accra", "Apam"]  # city_batch mode

    # ==================== PHASED PROCESSING ====================
    ENABLE_CHECKPOINT = True  # Track completed cities; resume after failure.
    STRICT_MODE = True  # Halt on any failure rather than auto-skipping.
    START_FROM_PHASE = 1  # Set to 2 or 3 to skip already-completed phases.

    # ==================== SKIP / OVERWRITE FLAGS ====================
    SKIP_BUILDING_EXTRACTION = False  # Must stay False for 3D metrics (FormFactor, H2F, VolDens).
    SKIP_METRIC_CALCULATION = False
    SKIP_STREET_NETWORK_CALCULATION = False
    SKIP_SPATIAL_DISTRIBUTION_CALCULATION = False
    SKIP_CONNECTIVITY_CALCULATION = False
    SKIP_COINS = False
    OVERWRITE_BUILDINGS = True
    OVERWRITE_GRID = True

    # ==================== FILE PATHS ====================
    # Companion grid-generation pipeline (africa_hex_grid_workflow):
    GRID_DATA_BASE = Path("/path/to/africa_hex_grid_workflow/data/03_country_grids")
    TIF_DATA_BASE = Path("/path/to/africa_hex_grid_workflow/data/00_country_TIF")
    # AFRICAPOLIS agglomeration boundaries and country/administrative boundaries
    # (see README "Data sources" - same sources as the companion grid pipeline):
    AGGLOMERATION_FILE = Path("/path/to/boundaries/agglomerations.gpkg")
    COUNTRY_BOUNDARIES_FILE = Path("/path/to/boundaries/africa_boundaries.gpkg")
    OUTPUT_BASE = Path(__file__).resolve().parent.parent  # repo root; per-city outputs land under OUTPUT_BASE/data/

    # ==================== OSM HIGHWAYS ====================
    OSM_FORCE_REDOWNLOAD = False
    OSM_OVERWRITE_EXISTING = False

    # ==================== PROCESSING PARAMETERS ====================
    CONFIDENCE_THRESHOLD = 0.75
    MIN_AREA_M2 = 8.0
    MAX_AREA_M2 = 100000.0
    NUM_WORKERS = min(16, max(4, cpu_count() * 2))
    EXPORT_FORMAT = "shapefile"
    GRID_ID_COLUMN = "GridID"
    GRID_CELL_SIZE = 250
    USE_SPATIAL_INDEX = True
    NUM_METRIC_WORKERS = None
    METRIC_CHUNK_SIZE = 15000
    CONNECTIVITY_BATCH_SIZE = 100
    EXPORT_CSV = True
    EXPORT_BUILDINGS_CSV = False
    EXPORT_GEOJSON = True

    # ==================== DYNAMIC PROPERTIES ====================
    @property
    def TARGET_CRS(self):
        """
        City-specific UTM CRS for the current ``CITY_NAME``.

        Each city is analyzed in its own locally-accurate UTM zone (see
        ``projection.py``) rather than one continental projection, to keep
        area/shape distortion negligible everywhere it matters. Falls back
        to ESRI:102022 (Africa Albers Equal Area Conic) only if the city or
        its country cannot be resolved in the projection mapping.
        """
        try:
            iso3 = self.get_iso3_for_city(self.CITY_NAME)
            if iso3:
                return get_crs_for_city(self.CITY_NAME, iso3)
            logger.warning(f"Could not determine country for {self.CITY_NAME}, using ESRI:102022 fallback")
            return "ESRI:102022"
        except Exception as e:
            logger.warning(f"Failed to get dynamic CRS for {self.CITY_NAME}: {e}, using ESRI:102022 fallback")
            return "ESRI:102022"

    @property
    def grid_boundary_file(self):
        country = self._get_country_for_city(self.CITY_NAME)
        if country is None:
            raise ValueError(f"Cannot detect country for city '{self.CITY_NAME}'")
        safe_city_name = sanitize_filename(self.CITY_NAME)
        return self.GRID_DATA_BASE / country / safe_city_name / f"{safe_city_name}_grid_boundary.geojson"

    @property
    def grid_file(self):
        country = self._get_country_for_city(self.CITY_NAME)
        if country is None:
            raise ValueError(f"Cannot detect country for city '{self.CITY_NAME}'")
        safe_city_name = sanitize_filename(self.CITY_NAME)
        return self.GRID_DATA_BASE / country / safe_city_name / f"{safe_city_name}_grid_250m_hex_enriched.shp"

    @property
    def GRID_BOUNDARY_FILE(self):
        return self.grid_boundary_file

    @property
    def GRID_FILE(self):
        return self.grid_file

    @property
    def highway_file(self):
        """
        Path to the city's clipped highway file.

        ``osm_network.process_osm_highways_for_cities`` writes to
        ``{OUTPUT_BASE}/data/{Country}/{City}/highways/{City}_highways.geojson``;
        this checks that path first, falling back to a couple of legacy
        layout patterns for compatibility with highway files produced by
        earlier standalone OSM extraction runs.
        """
        country = self._get_country_for_city(self.CITY_NAME)
        if country is None:
            return None

        safe_city_name = sanitize_filename(self.CITY_NAME)

        integrated_path = self.OUTPUT_BASE / "data" / country / safe_city_name / "highways" / f"{safe_city_name}_highways.geojson"
        if integrated_path.exists():
            return integrated_path

        legacy_highways_base = Path("/path/to/legacy/01_extracted/highways")
        legacy_patterns = [
            legacy_highways_base / f"{safe_city_name}_hwys.geojson",
        ]
        for pattern in legacy_patterns:
            if pattern.exists():
                return pattern

        return integrated_path  # Not yet created; caller decides how to handle a missing file.

    @property
    def height_raster_file(self):
        country = self._get_country_for_city(self.CITY_NAME)
        if country is None:
            return None
        height_path = self.TIF_DATA_BASE / country / "building_height.tif"
        return height_path if height_path.exists() else None

    @property
    def volume_raster_file(self):
        country = self._get_country_for_city(self.CITY_NAME)
        if country is None:
            return None
        volume_path = self.TIF_DATA_BASE / country / "building_volume.tif"
        return volume_path if volume_path.exists() else None

    @property
    def buildings_output_dir(self):
        country = self._get_country_for_city(self.CITY_NAME)
        if country is None:
            raise ValueError(f"Cannot detect country for city '{self.CITY_NAME}'")
        safe_city_name = sanitize_filename(self.CITY_NAME)
        return self.OUTPUT_BASE / f"data/{country}/{safe_city_name}/Output"

    @property
    def analysis_output_dir(self):
        return self.buildings_output_dir

    @property
    def BUILDINGS_OUTPUT_DIR(self):
        return self.buildings_output_dir

    @property
    def ANALYSIS_OUTPUT_DIR(self):
        return self.analysis_output_dir

    @property
    def enriched_grid_output_path(self):
        safe_city_name = sanitize_filename(self.CITY_NAME)
        return self.analysis_output_dir / f"{safe_city_name}_grid_250m_hex_enriched_metrics_v15.shp"

    # ==================== HELPERS ====================
    def get_cities_in_country(self, iso3_code):
        """List agglomeration names for a country, via the cached boundary data."""
        try:
            if not self.AGGLOMERATION_FILE.exists():
                return []
            agglom = get_agglomeration_data(self.AGGLOMERATION_FILE)
            if agglom is None:
                return []
            # Preserve original city names (including any incidental trailing
            # whitespace in the source data); sanitize_filename() handles that
            # when a name is turned into a path component.
            cities = agglom[agglom['ISO3'] == iso3_code]['Agglomeration_Name'].tolist()
            logger.info(f"Found {len(cities)} cities in {iso3_code}")
            return cities
        except Exception as e:
            logger.error(f"Error getting cities: {str(e)}")
            return []

    # A small number of countries have cities in the agglomeration file but
    # no matching entry in the country boundaries file (e.g. Djibouti); this
    # fills in the country name for those cases rather than dropping them.
    _ISO3_TO_COUNTRY_FALLBACK = {
        'DJI': 'Djibouti',
    }

    def _get_country_for_city(self, city_name):
        """Resolve a city name to a country name via ISO3, using cached lookups."""
        try:
            iso3_code = None

            if self.AGGLOMERATION_FILE.exists():
                agglom = get_agglomeration_data(self.AGGLOMERATION_FILE)
                if agglom is not None:
                    city_row = agglom[agglom['Agglomeration_Name'] == city_name]
                    if len(city_row) == 0:
                        city_row = agglom[agglom['Agglomeration_Name'].str.strip() == city_name.strip()]
                    if len(city_row) > 0:
                        iso3_code = city_row.iloc[0]['ISO3']

            if not iso3_code:
                return None

            if self.COUNTRY_BOUNDARIES_FILE.exists():
                countries = get_boundaries_data(self.COUNTRY_BOUNDARIES_FILE)
                if countries is not None:
                    country_row = countries[countries["ISO"] == iso3_code]
                    if len(country_row) > 0:
                        country_name = country_row.iloc[0]["NAME_0"]
                        return country_name.replace("'", "").replace("(", "").replace(")", "")

            if iso3_code in self._ISO3_TO_COUNTRY_FALLBACK:
                return self._ISO3_TO_COUNTRY_FALLBACK[iso3_code]

            return None
        except Exception:
            return None

    def get_iso3_for_city(self, city_name):
        """Look up a city's ISO3 country code from the cached agglomeration data."""
        try:
            if not self.AGGLOMERATION_FILE.exists():
                return None
            agglom = get_agglomeration_data(self.AGGLOMERATION_FILE)
            if agglom is None:
                return None
            city_row = agglom[agglom['Agglomeration_Name'] == city_name]
            if len(city_row) == 0:
                city_row = agglom[agglom['Agglomeration_Name'].str.strip() == city_name.strip()]
            if len(city_row) > 0:
                return city_row.iloc[0]['ISO3']
            return None
        except Exception:
            return None

    def get_country_name_from_iso3(self, iso3_code):
        try:
            if not self.COUNTRY_BOUNDARIES_FILE.exists():
                return None
            countries = get_boundaries_data(self.COUNTRY_BOUNDARIES_FILE)
            if countries is None:
                return None
            country_row = countries[countries["ISO"] == iso3_code]
            if len(country_row) > 0:
                return country_row.iloc[0]["NAME_0"]
            return None
        except Exception:
            return None

    # ==================== PHASED PROCESSING ====================
    def get_all_africa_cities(self):
        """All cities across Africa (including North Africa), for Phase 3."""
        try:
            if not self.AGGLOMERATION_FILE.exists():
                return []
            agglom = get_agglomeration_data(self.AGGLOMERATION_FILE)
            if agglom is None:
                return []
            africa_cities = agglom[agglom['ISO3'].isin(ALL_AFRICA_ISO3)]['Agglomeration_Name'].tolist()
            logger.info(f"Found {len(africa_cities)} cities in Africa (including North Africa)")
            return africa_cities
        except Exception as e:
            logger.error(f"Error getting Africa cities: {str(e)}")
            return []

    def get_all_sub_saharan_cities(self):
        """All Sub-Saharan African cities (kept for callers that need the older regional split)."""
        try:
            if not self.AGGLOMERATION_FILE.exists():
                return []
            agglom = get_agglomeration_data(self.AGGLOMERATION_FILE)
            if agglom is None:
                return []
            ssa_cities = agglom[agglom['ISO3'].isin(SUB_SAHARAN_AFRICA_ISO3)]['Agglomeration_Name'].tolist()
            logger.info(f"Found {len(ssa_cities)} cities in Sub-Saharan Africa")
            return ssa_cities
        except Exception as e:
            logger.error(f"Error getting Sub-Saharan Africa cities: {str(e)}")
            return []

    def get_phase1_cities(self):
        return PHASE1_SAMPLE_CITIES.copy()

    def get_phase2_cities(self):
        """Ghana cities not already covered by the Phase 1 sample."""
        try:
            ghana_cities = self.get_cities_in_country(PHASE2_COUNTRY_ISO3)
            phase2_cities = [city for city in ghana_cities if city not in PHASE2_EXCLUDE_CITIES]
            logger.info(f"Phase 2: {len(phase2_cities)} remaining Ghana cities (excluding {len(PHASE2_EXCLUDE_CITIES)} from Phase 1)")
            return phase2_cities
        except Exception as e:
            logger.error(f"Error getting Phase 2 cities: {str(e)}")
            return []

    def get_phase3_cities(self):
        """Every remaining African city not already covered by Phase 1 or 2."""
        try:
            all_africa_cities = self.get_all_africa_cities()
            phase1_cities = set(self.get_phase1_cities())
            phase2_cities = set(self.get_phase2_cities())
            phase3_cities = [
                city for city in all_africa_cities
                if city not in phase1_cities and city not in phase2_cities
            ]
            logger.info(f"Phase 3: {len(phase3_cities)} remaining African cities (including North Africa)")
            return phase3_cities
        except Exception as e:
            logger.error(f"Error getting Phase 3 cities: {str(e)}")
            return []

    def get_cities_for_phase(self, phase: int):
        if phase == 1:
            return self.get_phase1_cities()
        elif phase == 2:
            return self.get_phase2_cities()
        elif phase == 3:
            return self.get_phase3_cities()
        else:
            logger.error(f"Invalid phase: {phase}")
            return []

    def validate_paths(self):
        """
        Check that a city has the inputs required to proceed.

        Returns (errors, is_too_small):
            errors: messages describing missing required inputs.
            is_too_small: True if the city's AOI is small enough that the
                companion grid pipeline produced zero hex cells for it (not
                an error - just too small an area for grid-based analysis).

        A missing highway file is not an error: it degrades gracefully by
        disabling street-network metrics for that city (expected for very
        small cities whose clipped OSM network is empty).
        """
        errors = []
        is_too_small = False

        if not self.SKIP_METRIC_CALCULATION and not self.grid_file.exists():
            errors.append(f"Grid file not found for {self.CITY_NAME}")

        if not self.SKIP_METRIC_CALCULATION and not self.grid_boundary_file.exists():
            if self.grid_file.exists():
                try:
                    grid_gdf = gpd.read_file(self.grid_file)
                    if len(grid_gdf) == 0:
                        logger.warning(f"{self.CITY_NAME}: AOI too small for grid analysis (no hex cell centers within AOI boundary)")
                        is_too_small = True
                    else:
                        errors.append(f"Grid boundary not found for {self.CITY_NAME} (but grid file exists with {len(grid_gdf)} cells)")
                except Exception as e:
                    logger.warning(f"Could not check grid file for {self.CITY_NAME}: {e}")
                    errors.append(f"Grid boundary not found for {self.CITY_NAME} (but grid file exists)")

        if not self.SKIP_STREET_NETWORK_CALCULATION and not self.highway_file.exists():
            logger.warning(f"{self.CITY_NAME}: No highway data after clipping (city too small) - skipping street network metrics")
            self.SKIP_STREET_NETWORK_CALCULATION = True

        return errors, is_too_small

    def create_output_dirs(self):
        for directory in [self.buildings_output_dir, self.analysis_output_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def check_grid_inputs(self):
        if self.grid_boundary_file.exists() and self.grid_file.exists():
            logger.info(f"Grid exists for {self.CITY_NAME}")
            return True
        logger.warning(f"Required grid or boundary not found for {self.CITY_NAME}")
        return False


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('building_analysis.log')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def load_grid_boundary(config):
    try:
        if not config.GRID_BOUNDARY_FILE.exists():
            return {"success": False, "message": "Boundary file not found"}
        gpd.read_file(config.GRID_BOUNDARY_FILE)
        return {"success": True, "file": config.GRID_BOUNDARY_FILE}
    except Exception as e:
        return {"success": False, "message": str(e)}


def extract_buildings_from_gcs(config, aoi_file, max_retries=1):
    """
    Extract buildings from Google Open Buildings via GCS, with retry.

    A zero-building result is retried up to ``max_retries`` times before
    being reported as a genuine no-data case (``no_data=True``) rather than
    an error, since Open Buildings coverage gaps are expected in some
    regions and are handled as a data-coverage condition, not a failure.

    Returns
    -------
    dict with keys: success, file, count, message, no_data, retry_count.
    """
    formats = {"geojson": ".geojson", "shapefile": ".shp", "csv": ".csv"}
    ext = formats.get(config.EXPORT_FORMAT, ".geojson")
    output_file = config.BUILDINGS_OUTPUT_DIR / f"open_buildings{ext}"

    if output_file.exists() and not config.OVERWRITE_BUILDINGS:
        if config.EXPORT_FORMAT != "csv":
            try:
                buildings = gpd.read_file(output_file)
                building_count = len(buildings)
                if building_count > 0:
                    logger.info(f"Using existing buildings file: {building_count:,} buildings")
                    return {"success": True, "file": output_file, "count": building_count,
                           "no_data": False, "retry_count": 0}
                else:
                    logger.warning("Existing file has 0 buildings, will re-attempt extraction")
            except Exception as e:
                logger.warning(f"Failed to read existing file: {e}, will re-attempt extraction")

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"Retry attempt {attempt}/{max_retries} for building extraction...")

            gcs_config = OpenBuildingsGCSConfig(
                aoi_file=aoi_file,
                output_dir=config.BUILDINGS_OUTPUT_DIR,
                confidence_threshold=config.CONFIDENCE_THRESHOLD,
                min_area_m2=config.MIN_AREA_M2,
                max_area_m2=config.MAX_AREA_M2,
                export_format=config.EXPORT_FORMAT,
                overwrite_existing=config.OVERWRITE_BUILDINGS or attempt > 0,
                num_workers=config.NUM_WORKERS
            )
            processor = OpenBuildingsGCSProcessor(gcs_config)
            result = processor.process()

            if result.success:
                building_count = result.processed_count
                logger.info(f"Extracted {building_count:,} buildings in {result.elapsed_time:.1f}s")

                if building_count == 0:
                    logger.warning(f"Zero buildings extracted for {config.CITY_NAME}")
                    if attempt >= max_retries:
                        logger.error(f"No buildings found after {attempt + 1} attempts")
                        return {
                            "success": False,
                            "message": f"Zero buildings found after {attempt + 1} attempts",
                            "no_data": True,
                            "retry_count": attempt,
                            "count": 0
                        }
                    logger.info(f"Will retry extraction (attempt {attempt + 1}/{max_retries})...")
                    continue

                if result.output_paths and len(result.output_paths) > 0:
                    output_path = result.output_paths[0]
                    if not output_path.exists():
                        logger.error(f"Output file not created: {output_path}")
                        return {
                            "success": False,
                            "message": "Output file not created",
                            "no_data": False,
                            "retry_count": attempt
                        }
                    return {
                        "success": True,
                        "file": output_path,
                        "count": building_count,
                        "no_data": False,
                        "retry_count": attempt
                    }
                else:
                    logger.error("No output paths returned from processor")
                    return {
                        "success": False,
                        "message": "No output paths returned",
                        "no_data": False,
                        "retry_count": attempt
                    }
            else:
                if attempt >= max_retries:
                    return {
                        "success": False,
                        "message": result.message,
                        "no_data": False,
                        "retry_count": attempt
                    }
                logger.warning(f"Extraction failed: {result.message}, will retry...")

        except Exception as e:
            logger.error(f"Exception during extraction (attempt {attempt + 1}): {str(e)}")
            if attempt >= max_retries:
                return {
                    "success": False,
                    "message": str(e),
                    "no_data": False,
                    "retry_count": attempt
                }
            logger.info("Will retry after exception...")

    return {
        "success": False,
        "message": "Extraction failed after all retries",
        "no_data": False,
        "retry_count": max_retries
    }


def calculate_street_network_metrics_for_grid(config, grid, buildings=None):
    if config.SKIP_STREET_NETWORK_CALCULATION:
        return grid
    try:
        highways = gpd.read_file(config.highway_file)
        grid, nodes_gdf, edges_gdf, graph = calculate_street_network_metrics_with_graph(
            roads=highways, cells=grid, buildings=buildings,
            cell_id_col=config.GRID_ID_COLUMN, area_col="Area_m2", target_crs=config.TARGET_CRS
        )
        config._street_nodes = nodes_gdf
        config._street_edges = edges_gdf
        config._street_graph = graph
        return grid
    except Exception as e:
        logger.error(f"Street network error: {str(e)}")
        config._street_nodes = None
        config._street_edges = None
        config._street_graph = None
        return grid


def calculate_connectivity_metrics_for_grid(config, grid):
    if config.SKIP_CONNECTIVITY_CALCULATION or not hasattr(config, '_street_graph') or config._street_graph is None:
        return grid
    try:
        spatial_graphs_files = list(config.ANALYSIS_OUTPUT_DIR.glob("spatial_graphs_*.pkl"))
        spatial_graphs_pkl = spatial_graphs_files[0] if spatial_graphs_files else config.ANALYSIS_OUTPUT_DIR / "dummy.pkl"
        grid = calculate_connectivity_metrics(
            grid=grid, nodes_gdf=config._street_nodes, edges_gdf=config._street_edges,
            graph=config._street_graph, spatial_graphs_pkl=spatial_graphs_pkl,
            grid_id_col=config.GRID_ID_COLUMN, skip_coins=config.SKIP_COINS,
            batch_size=config.CONNECTIVITY_BATCH_SIZE
        )
        return grid
    except Exception as e:
        logger.error(f"Connectivity error: {str(e)}")
        return grid


def calculate_and_aggregate_metrics(config, buildings_file):
    """Calculate building/spatial/street/connectivity metrics and aggregate onto the grid."""
    if config.SKIP_METRIC_CALCULATION:
        return None, None

    if config.enriched_grid_output_path.exists() and not config.OVERWRITE_GRID:
        response = input("Overwrite existing enriched grid? (y/n): ").strip().lower()
        if response != 'y':
            return gpd.read_file(config.enriched_grid_output_path), gpd.read_file(buildings_file)

    try:
        processing_steps = ["Loading buildings", "Loading grid", "Reprojecting data", "Calculating building metrics",
                          "Aggregating to grid", "Calculating spatial distribution metrics",
                          "Calculating street network metrics", "Calculating connectivity metrics", "Saving enriched grid"]

        with tqdm(total=len(processing_steps), desc="Processing", unit="step", position=0) as main_pbar:
            main_pbar.set_postfix_str("Loading buildings")
            buildings = gpd.read_file(buildings_file)
            if config.EXPORT_BUILDINGS_CSV:
                buildings_csv = buildings_file.with_suffix('.csv')
                if not buildings_csv.exists():
                    buildings[[col for col in buildings.columns if col != 'geometry']].to_csv(buildings_csv, index=False)
            main_pbar.update(1)

            main_pbar.set_postfix_str("Loading grid")
            grid = gpd.read_file(config.GRID_FILE)
            main_pbar.update(1)

            main_pbar.set_postfix_str("Reprojecting")
            buildings = buildings.to_crs(config.TARGET_CRS)
            grid = grid.to_crs(buildings.crs)
            if config.USE_SPATIAL_INDEX:
                buildings.sindex
                grid.sindex
            main_pbar.update(1)

            main_pbar.set_postfix_str("Calculating metrics")
            buildings = calculate_building_metrics(buildings, n_jobs=config.NUM_METRIC_WORKERS,
                                                  chunk_size=config.METRIC_CHUNK_SIZE,
                                                  height_raster_path=config.height_raster_file,
                                                  volume_raster_path=config.volume_raster_file)
            main_pbar.update(1)

            main_pbar.set_postfix_str("Aggregating to grid")
            grid = aggregate_metrics_to_grid(buildings, grid, grid_id_col=config.GRID_ID_COLUMN)
            main_pbar.update(1)

            main_pbar.set_postfix_str("Spatial distribution metrics")
            if not config.SKIP_SPATIAL_DISTRIBUTION_CALCULATION:
                try:
                    # grid already carries every previously merged column, so
                    # this reassignment keeps them rather than dropping to a
                    # smaller "only the new columns" frame.
                    buildings, grid = calculate_and_aggregate_spatial_metrics(
                        buildings, grid, grid_id_col=config.GRID_ID_COLUMN, target_crs=config.TARGET_CRS,
                        k=5, orientation_col='Orientation')
                except Exception as e:
                    logger.error(f"Spatial distribution calculation failed: {str(e)}")
            main_pbar.update(1)

            main_pbar.set_postfix_str("Street network metrics")
            if config.GRID_ID_COLUMN not in buildings.columns:
                buildings_with_grid = gpd.sjoin(buildings, grid[[config.GRID_ID_COLUMN, 'geometry']],
                                                predicate='within', how='left')
                buildings = buildings.copy()
                buildings[config.GRID_ID_COLUMN] = buildings_with_grid[config.GRID_ID_COLUMN]
            grid = calculate_street_network_metrics_for_grid(config, grid, buildings)
            main_pbar.update(1)

            main_pbar.set_postfix_str("Connectivity metrics")
            grid = calculate_connectivity_metrics_for_grid(config, grid)
            main_pbar.update(1)

            main_pbar.set_postfix_str("Saving results")
            grid.to_file(config.enriched_grid_output_path)
            if config.EXPORT_CSV:
                grid[[col for col in grid.columns if col != 'geometry']].to_csv(
                    config.enriched_grid_output_path.with_suffix('.csv'), index=False)
            if config.EXPORT_GEOJSON:
                grid.to_file(config.enriched_grid_output_path.with_suffix('.geojson'), driver='GeoJSON')
            main_pbar.update(1)

        return grid, buildings
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
        raise


def print_summary(config, boundary_result, buildings_result, grid, buildings, elapsed_time):
    logger.info("\n" + "=" * 80)
    logger.info(f"SUMMARY - {config.CITY_NAME}")
    logger.info("=" * 80)
    logger.info(f"Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
    if buildings_result["success"]:
        logger.info(f"Buildings: {buildings_result['count']:,}")
    if grid is not None and 'Mean_Area' in grid.columns:
        logger.info(f"Grid cells: {grid['Mean_Area'].notna().sum():,} / {len(grid):,}")
    logger.info(f"Output: {config.ANALYSIS_OUTPUT_DIR}")
    logger.info("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def process_single_city(config, city_name):
    """Process one city end to end: extract buildings, calculate and aggregate metrics."""
    city_start_time = time.time()
    config.CITY_NAME = city_name

    # Reset per-city, since STRICT_MODE processing reuses one Config object
    # across many cities and this flag can be set True by validate_paths().
    config.SKIP_STREET_NETWORK_CALCULATION = False

    errors, is_too_small = config.validate_paths()
    if is_too_small:
        logger.warning(f"{city_name}: Skipping - AOI too small for grid analysis")
        return {
            "success": False,
            "city": city_name,
            "error": "AOI too small for grid analysis",
            "too_small": True
        }
    if errors:
        logger.error(f"Path validation failed for {city_name}: {errors}")
        return {"success": False, "city": city_name, "error": "Path validation failed"}

    config.create_output_dirs()
    boundary_result = load_grid_boundary(config)
    if not boundary_result["success"]:
        return {"success": False, "city": city_name, "error": "Grid boundary loading failed"}

    buildings_result = extract_buildings_from_gcs(config, boundary_result["file"])
    if not buildings_result["success"]:
        if buildings_result.get("no_data", False):
            return {
                "success": False,
                "city": city_name,
                "error": "No building data available",
                "no_data": True,
                "retry_count": buildings_result.get("retry_count", 0)
            }
        return {"success": False, "city": city_name, "error": "Building extraction failed"}

    grid, buildings = calculate_and_aggregate_metrics(config, buildings_result["file"])
    elapsed_time = time.time() - city_start_time
    print_summary(config, boundary_result, buildings_result, grid, buildings, elapsed_time)

    return {"success": True, "city": city_name, "elapsed_time": elapsed_time,
            "building_count": buildings_result.get("count", 0)}


def check_grid_inputs_batch(config, cities):
    """Select cities with both required companion-grid input files."""
    logger.info("=" * 80 + f"\nPHASE 1: Grid Input Check ({len(cities)} cities)\n" + "=" * 80)
    successful_cities = []
    failed_cities = []

    for i, city in enumerate(cities, 1):
        original_city = config.CITY_NAME
        config.CITY_NAME = city
        try:
            if config.grid_boundary_file.exists() and config.grid_file.exists():
                logger.info(f"[{i}/{len(cities)}] {city}: Grid exists, skipping")
                successful_cities.append(city)
            else:
                logger.error(f"[{i}/{len(cities)}] {city}: Required grid input missing")
                failed_cities.append(city)
        except Exception as e:
            logger.error(f"[{i}/{len(cities)}] {city}: Error: {str(e)}")
            failed_cities.append(city)
        finally:
            config.CITY_NAME = original_city

    logger.info(f"\nPhase 1: {len(successful_cities)}/{len(cities)} successful")
    if failed_cities:
        logger.warning(f"Failed: {failed_cities}")

    return {"success": len(failed_cities) == 0, "successful_cities": successful_cities, "failed_cities": failed_cities}


def run_osm_highways_batch(config, cities):
    """Download country-level OSM data (once per country) and clip highways for a batch of cities."""
    logger.info("\n" + "=" * 80 + f"\nPHASE 1.5: OSM Highway Download & Clip ({len(cities)} cities)\n" + "=" * 80)

    try:
        highway_results = process_osm_highways_for_cities(
            city_names=cities,
            base_output_dir=config.OUTPUT_BASE / "data",
            agglomeration_file=config.AGGLOMERATION_FILE,
            grid_data_base=config.GRID_DATA_BASE,
            country_boundaries_file=config.COUNTRY_BOUNDARIES_FILE,
            logger=logger,
            force_redownload=config.OSM_FORCE_REDOWNLOAD,
            overwrite_existing=config.OSM_OVERWRITE_EXISTING
        )

        successful = [city for city, path in highway_results.items() if path is not None]
        failed = [city for city, path in highway_results.items() if path is None]

        logger.info(f"\nPhase 1.5: {len(successful)}/{len(cities)} successful")
        if failed:
            logger.warning(f"Failed or no highways: {failed}")

        return {
            "success": len(successful) > 0,
            "successful_cities": successful,
            "failed_cities": failed,
            "highway_results": highway_results
        }
    except Exception as e:
        logger.error(f"OSM highways processing failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "successful_cities": [],
            "failed_cities": cities,
            "highway_results": {}
        }


def run_metrics_batch(config, cities):
    """Run metrics calculation for a batch of cities."""
    logger.info("\n" + "=" * 80 + f"\nPHASE 2: Metrics Calculation ({len(cities)} cities)\n" + "=" * 80)
    results = []
    successful = 0
    failed = 0

    for i, city in enumerate(cities, 1):
        logger.info(f"\n[{i}/{len(cities)}] {city}")
        result = process_single_city(config, city)
        results.append(result)
        if result["success"]:
            successful += 1
        else:
            failed += 1

    logger.info("\n" + "=" * 80 + f"\nPhase 2: {successful}/{len(cities)} successful")
    if successful > 0:
        total_buildings = sum(r.get("building_count", 0) for r in results if r["success"])
        total_time = sum(r.get("elapsed_time", 0) for r in results if r["success"])
        logger.info(f"Buildings: {total_buildings:,}, Time: {total_time:.1f}s ({total_time/60:.1f}m)")
    if failed > 0:
        logger.warning(f"Failed: {[r['city'] for r in results if not r['success']]}")

    return {"success": failed == 0, "results": results, "successful": successful, "failed": failed}


def main():
    """Entry point. Supports single_city, city_batch, country_batch, and phased_automatic modes."""
    start_time = time.time()
    logger.info("=" * 80 + "\nBUILDING MORPHOLOGY ANALYSIS PIPELINE\n" + "=" * 80)

    if not USING_IGRAPH:
        logger.warning("igraph unavailable, using NetworkX connectivity backend (slower)")

    config = Config()
    email_notifier = EmailNotifier(logger=logger)

    checkpoint_dir = Path(__file__).resolve().parent.parent / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    no_data_tracker = NoBuildingDataTracker(checkpoint_dir / "no_building_data_cities.json")
    excluded_countries_tracker = ExcludedCountriesTracker(checkpoint_dir / "excluded_countries.json")
    excluded_countries_tracker.initialize_active_countries(ALL_AFRICA_ISO3)

    if config.PROCESSING_MODE == "single_city":
        logger.info(f"Mode: Single City - {config.CITY_NAME}")

        if not config.check_grid_inputs():
            logger.error("Grid input files unavailable. Supply the companion grid outputs first.")
            sys.exit(1)

        if not config.SKIP_STREET_NETWORK_CALCULATION:
            osm_result = run_osm_highways_batch(config, [config.CITY_NAME])
            if not osm_result["success"]:
                logger.warning("OSM highway download failed, continuing without street network metrics")

        result = process_single_city(config, config.CITY_NAME)
        if not result["success"]:
            error_msg = f"Processing failed: {result.get('error')}"
            logger.error(error_msg)
            email_notifier.send_error_notification(
                error_message=error_msg,
                city_name=config.CITY_NAME,
                context={'mode': 'single_city'}
            )
            sys.exit(1)
        else:
            elapsed_time = time.time() - start_time
            email_notifier.send_task_completion(
                mode='single_city',
                summary_stats={
                    'City': config.CITY_NAME,
                    'Buildings': result.get('building_count', 0),
                    'Processing Time': f"{elapsed_time:.1f}s ({elapsed_time/60:.1f}m)",
                    'Status': 'SUCCESS'
                }
            )

    elif config.PROCESSING_MODE == "city_batch":
        logger.info("Mode: City Batch - Manual List")
        cities = config.CITY_LIST
        if not cities:
            logger.error("CITY_LIST is empty. Please define cities in the configuration.")
            sys.exit(1)
        logger.info(f"Processing {len(cities)} cities: {cities}")

        grid_results = check_grid_inputs_batch(config, cities)
        if not grid_results["successful_cities"]:
            logger.error("No valid grids. Cannot proceed.")
            sys.exit(1)

        if not config.SKIP_STREET_NETWORK_CALCULATION:
            run_osm_highways_batch(config, grid_results["successful_cities"])

        metrics_results = run_metrics_batch(config, grid_results["successful_cities"])

        elapsed_time = time.time() - start_time
        status = 'SUCCESS' if metrics_results["success"] else 'PARTIAL SUCCESS (some failures)'
        if not metrics_results["success"]:
            logger.warning("Some cities failed during metrics calculation")
        email_notifier.send_task_completion(
            mode='city_batch',
            summary_stats={
                'Total Cities': len(cities),
                'Successful': metrics_results['successful'],
                'Failed': metrics_results['failed'],
                'Processing Time': f"{elapsed_time/3600:.2f} hours",
                'Status': status
            }
        )

    elif config.PROCESSING_MODE == "country_batch":
        logger.info(f"Mode: Country Batch - {config.TARGET_COUNTRY_ISO3}")
        cities = config.get_cities_in_country(config.TARGET_COUNTRY_ISO3)
        if not cities:
            logger.error(f"No cities found for {config.TARGET_COUNTRY_ISO3}")
            sys.exit(1)
        logger.info(f"Found {len(cities)} cities")

        grid_results = check_grid_inputs_batch(config, cities)
        if not grid_results["successful_cities"]:
            logger.error("No valid grids. Cannot proceed.")
            sys.exit(1)

        if not config.SKIP_STREET_NETWORK_CALCULATION:
            run_osm_highways_batch(config, grid_results["successful_cities"])

        metrics_results = run_metrics_batch(config, grid_results["successful_cities"])

        elapsed_time = time.time() - start_time
        country_name = config.get_country_name_from_iso3(config.TARGET_COUNTRY_ISO3) or config.TARGET_COUNTRY_ISO3
        status = 'SUCCESS' if metrics_results["success"] else 'PARTIAL SUCCESS (some failures)'
        if not metrics_results["success"]:
            logger.warning("Some cities failed during metrics calculation")
        email_notifier.send_task_completion(
            mode='country_batch',
            summary_stats={
                'Country': f"{country_name} ({config.TARGET_COUNTRY_ISO3})",
                'Total Cities': len(cities),
                'Successful': metrics_results['successful'],
                'Failed': metrics_results['failed'],
                'Processing Time': f"{elapsed_time/3600:.2f} hours",
                'Status': status
            }
        )

    elif config.PROCESSING_MODE == "phased_automatic":
        logger.info("=" * 80)
        logger.info("Mode: PHASED AUTOMATIC PROCESSING")
        logger.info("  Phase 1: sample cities (all of Africa)")
        logger.info("  Phase 2: remaining Ghana cities")
        logger.info("  Phase 3: all remaining African cities (including North Africa)")
        logger.info(f"  Strict Mode: {config.STRICT_MODE} (halt on any failure)")
        logger.info(f"  Checkpoint Enabled: {config.ENABLE_CHECKPOINT}")
        logger.info("=" * 80)

        checkpoint = None
        if config.ENABLE_CHECKPOINT:
            checkpoint_path = get_checkpoint_path(config.OUTPUT_BASE)
            checkpoint = ProcessingCheckpoint(checkpoint_path)
            checkpoint.print_statistics()

            # Seed no_data_tracker's per-country success state from the
            # checkpoint so a resumed run doesn't wrongly treat a country
            # as eligible for exclusion just because its earlier successes
            # happened in a prior process.
            logger.info("Initializing country success tracking from checkpoint...")
            completed_countries = set()
            for city, details in checkpoint.data.get('city_details', {}).items():
                if details.get('status') == 'success' and 'iso3' in details:
                    iso3 = details['iso3']
                    if iso3 not in completed_countries:
                        completed_countries.add(iso3)
                        no_data_tracker.reset_country_consecutive_failures(iso3)
                        logger.info(f"  Marked {iso3} as having successful cities (from checkpoint)")
            if completed_countries:
                logger.info(f"Loaded {len(completed_countries)} countries with successful cities from checkpoint")
            else:
                logger.info("No previously successful countries found in checkpoint")

        for phase in [1, 2, 3]:
            if phase < config.START_FROM_PHASE:
                logger.info("\n" + "=" * 80)
                logger.info(f"SKIPPING PHASE {phase} (START_FROM_PHASE={config.START_FROM_PHASE})")
                logger.info("=" * 80)
                continue

            phase_start_time = time.time()
            logger.info("\n" + "=" * 80)
            logger.info(f"STARTING PHASE {phase}")
            logger.info("=" * 80)

            all_phase_cities = config.get_cities_for_phase(phase)
            if not all_phase_cities:
                logger.warning(f"Phase {phase}: No cities to process, skipping")
                continue

            if checkpoint:
                pending_cities = checkpoint.get_pending_cities(all_phase_cities, phase)
                completed_count = len(all_phase_cities) - len(pending_cities)
                logger.info(f"Phase {phase}: {len(all_phase_cities)} total cities")
                logger.info(f"  Already completed: {completed_count}")
                logger.info(f"  Pending: {len(pending_cities)}")
            else:
                pending_cities = all_phase_cities
                logger.info(f"Phase {phase}: {len(pending_cities)} cities to process")

            if not pending_cities:
                logger.info(f"Phase {phase} already completed, moving to next phase")
                continue

            phase_successful = []
            phase_failed = []

            # For Phase 2 & 3, track per-country completion so a completion
            # email can be sent once every city in a country has resolved.
            country_tracker = {} if phase in [2, 3] else None
            if phase in [2, 3]:
                for city in all_phase_cities:
                    iso3 = config.get_iso3_for_city(city)
                    if iso3:
                        if iso3 not in country_tracker:
                            country_tracker[iso3] = {'total': [], 'completed': [], 'failed': [], 'start_time': None}
                        country_tracker[iso3]['total'].append(city)
                        if checkpoint:
                            city_status = checkpoint.get_city_status(city, phase)
                            if city_status == 'success':
                                country_tracker[iso3]['completed'].append(city)
                            elif city_status == 'failed':
                                country_tracker[iso3]['failed'].append(city)

                if phase == 3 and country_tracker:
                    already_completed_countries = []
                    for iso3, country_data in country_tracker.items():
                        total_count = len(country_data['total'])
                        completed_count = len(country_data['completed'])
                        failed_count = len(country_data['failed'])
                        if total_count > 0 and (completed_count + failed_count == total_count):
                            country_name = config.get_country_name_from_iso3(iso3) or iso3
                            logger.info(f"\n{'='*80}")
                            logger.info(f"COUNTRY ALREADY COMPLETED (from checkpoint): {country_name} ({iso3})")
                            logger.info(f"  Total cities: {total_count}")
                            logger.info(f"  Successful: {completed_count}")
                            logger.info(f"  Failed: {failed_count}")
                            logger.info(f"{'='*80}\n")
                            email_notifier.send_country_completion(
                                country_name=country_name, iso3=iso3, phase=phase,
                                total_cities=total_count, successful_cities=country_data['completed'],
                                failed_cities=country_data['failed'], total_time=0.0
                            )
                            already_completed_countries.append(iso3)
                    if already_completed_countries:
                        logger.info(f"Sent completion emails for {len(already_completed_countries)} previously completed countries:")
                        for iso3 in already_completed_countries:
                            country_name = config.get_country_name_from_iso3(iso3) or iso3
                            logger.info(f"  - {country_name} ({iso3})")

            for i, city_name in enumerate(pending_cities, 1):
                logger.info("\n" + "-" * 80)
                logger.info(f"Phase {phase} - City {i}/{len(pending_cities)}: {city_name}")
                logger.info("-" * 80)

                config.CITY_NAME = city_name
                iso3 = config.get_iso3_for_city(city_name)
                if iso3 is None:
                    logger.warning(f"Cannot find ISO3 code for {city_name}, using 'UNKNOWN'")
                    iso3 = "UNKNOWN"

                if excluded_countries_tracker.is_country_excluded(iso3):
                    country_name = config.get_country_name_from_iso3(iso3) or iso3
                    logger.info(f"Skipping {city_name} - Country {country_name} ({iso3}) excluded (no Open Buildings data)")
                    phase_failed.append(city_name)
                    if phase == 3 and country_tracker and iso3 in country_tracker:
                        country_tracker[iso3]['failed'].append(city_name)
                    if checkpoint:
                        checkpoint.mark_city_failed(city_name, phase, iso3, "Country excluded - no Open Buildings data")
                    continue

                country_name = config.get_country_name_from_iso3(iso3) or iso3
                if no_data_tracker.is_known_no_data_city(city_name, country_name):
                    logger.info(f"Skipping {city_name} - Already known to have no building data (from previous run)")
                    phase_failed.append(city_name)
                    if phase == 3 and country_tracker and iso3 in country_tracker:
                        country_tracker[iso3]['failed'].append(city_name)
                    if checkpoint:
                        checkpoint.mark_city_failed(city_name, phase, iso3, "Known no building data (skipped)")
                    continue

                if phase == 3 and country_tracker and iso3 in country_tracker:
                    if country_tracker[iso3]['start_time'] is None:
                        country_tracker[iso3]['start_time'] = time.time()

                try:
                    if not config.check_grid_inputs():
                        error_msg = f"Grid generation failed for {city_name}"
                        logger.error(error_msg)
                        phase_failed.append(city_name)
                        if phase == 3 and country_tracker and iso3 in country_tracker:
                            country_tracker[iso3]['failed'].append(city_name)
                        if checkpoint:
                            checkpoint.mark_city_failed(city_name, phase, iso3, error_msg)
                        email_notifier.send_error_notification(
                            error_message=error_msg, city_name=city_name, phase=phase,
                            context={'iso3': iso3, 'step': 'Grid Generation'}
                        )
                        if config.STRICT_MODE:
                            logger.error(f"STRICT MODE: Halting due to failure in {city_name}")
                            logger.error("Fix the error and restart the pipeline to resume from checkpoint")
                            sys.exit(1)
                        continue

                    if not config.SKIP_STREET_NETWORK_CALCULATION:
                        osm_result = run_osm_highways_batch(config, [city_name])
                        if not osm_result["success"]:
                            error_msg = f"OSM highway download failed for {city_name}"
                            city_name_clean = city_name.strip()
                            is_coverage_gap = (
                                'highway_results' in osm_result and
                                city_name_clean in osm_result['highway_results'] and
                                osm_result['highway_results'][city_name_clean] is None
                            )
                            if is_coverage_gap:
                                logger.warning(f"{city_name}: No OSM highways available (coverage gap)")
                                logger.info("  Continuing with metrics calculation (street network metrics will be skipped)")
                            else:
                                logger.error(error_msg)
                                phase_failed.append(city_name)
                                if phase == 3 and country_tracker and iso3 in country_tracker:
                                    country_tracker[iso3]['failed'].append(city_name)
                                if checkpoint:
                                    checkpoint.mark_city_failed(city_name, phase, iso3, error_msg)
                                email_notifier.send_error_notification(
                                    error_message=error_msg, city_name=city_name, phase=phase,
                                    context={'iso3': iso3, 'step': 'OSM Highways'}
                                )
                                if config.STRICT_MODE:
                                    logger.error(f"STRICT MODE: Halting due to OSM failure in {city_name}")
                                    logger.error("Fix the error and restart the pipeline to resume from checkpoint")
                                    sys.exit(1)
                                continue

                    result = process_single_city(config, city_name)
                    if not result["success"]:
                        if result.get("too_small", False):
                            logger.warning(f"{city_name}: AOI too small for grid analysis - skipping")
                            error_msg = f"AOI too small for grid analysis: {city_name}"
                            phase_failed.append(city_name)
                            if phase == 3 and country_tracker and iso3 in country_tracker:
                                country_tracker[iso3]['failed'].append(city_name)
                            if checkpoint:
                                checkpoint.mark_city_failed(city_name, phase, iso3, error_msg)
                            logger.info(f"Skipping {city_name} (too small) and continuing with next city...")
                            continue

                        if result.get("no_data", False):
                            logger.warning(f"{city_name}: No building data available (Open Buildings coverage gap)")
                            try:
                                boundary = gpd.read_file(config.grid_boundary_file)
                                boundary_wgs84 = boundary.to_crs('EPSG:4326')
                                bounds = boundary_wgs84.total_bounds
                                area_km2 = boundary.geometry.area.sum() / 1e6
                                coordinates = {
                                    "lon_min": float(bounds[0]), "lat_min": float(bounds[1]),
                                    "lon_max": float(bounds[2]), "lat_max": float(bounds[3])
                                }
                            except Exception as e:
                                logger.warning(f"Could not extract boundary info: {e}")
                                area_km2 = None
                                coordinates = None

                            country_name = config.get_country_name_from_iso3(iso3) or iso3
                            no_data_tracker.add_city(
                                city_name=city_name, country_name=country_name, iso3=iso3,
                                retry_count=result.get("retry_count", 0), boundary_area_km2=area_km2,
                                coordinates=coordinates,
                                notes=f"Zero buildings found after {result.get('retry_count', 0) + 1} attempts"
                            )
                            email_notifier.send_no_building_data_notification(
                                city_name=city_name, country_name=country_name, iso3=iso3,
                                retry_count=result.get('retry_count', 0), coordinates=coordinates, area_km2=area_km2
                            )

                            exclusion_check = no_data_tracker.check_country_consecutive_failures(iso3, city_name)
                            if "reason" in exclusion_check and exclusion_check["reason"]:
                                logger.info(f"  Exclusion check for {iso3}: {exclusion_check['reason']}")

                            if exclusion_check["should_exclude"] and not excluded_countries_tracker.is_country_excluded(iso3):
                                try:
                                    country_cities = config.get_cities_in_country(iso3)
                                    total_cities_in_country = len(country_cities)
                                except Exception:
                                    total_cities_in_country = 0

                                excluded_countries_tracker.exclude_country(
                                    iso3=iso3, country_name=country_name,
                                    failed_cities=exclusion_check["failed_cities"],
                                    total_cities_in_country=total_cities_in_country,
                                    reason=f"Consecutive {exclusion_check['consecutive_count']} cities with no Open Buildings data"
                                )

                                remaining_countries = excluded_countries_tracker.get_remaining_country_count()
                                excluded_countries = excluded_countries_tracker.get_excluded_countries()
                                excluded_country_names = [info['country_name'] for info in excluded_countries.values()]

                                try:
                                    agglomeration = get_agglomeration_data(config.AGGLOMERATION_FILE)
                                    active_iso3_list = excluded_countries_tracker.get_active_countries_iso3()
                                    remaining_cities_count = len(agglomeration[agglomeration['ISO3'].isin(active_iso3_list)])
                                except Exception:
                                    remaining_cities_count = 0

                                email_notifier.send_country_excluded_notification(
                                    country_name=country_name, iso3=iso3,
                                    failed_cities=exclusion_check["failed_cities"],
                                    total_cities_in_country=total_cities_in_country,
                                    remaining_countries_count=remaining_countries,
                                    remaining_cities_count=remaining_cities_count,
                                    excluded_countries_list=excluded_country_names
                                )

                                logger.warning("=" * 80)
                                logger.warning(f"COUNTRY EXCLUDED: {country_name} ({iso3})")
                                logger.warning(f"  Reason: {exclusion_check['consecutive_count']} consecutive cities with no building data")
                                logger.warning(f"  Failed cities: {', '.join(exclusion_check['failed_cities'])}")
                                logger.warning(f"  Total cities in country: {total_cities_in_country}")
                                logger.warning(f"  Remaining active countries: {remaining_countries}")
                                logger.warning(f"  Remaining total cities: {remaining_cities_count}")
                                logger.warning("=" * 80)

                            error_msg = f"No building data available for {city_name}"
                            phase_failed.append(city_name)
                            if phase == 3 and country_tracker and iso3 in country_tracker:
                                country_tracker[iso3]['failed'].append(city_name)
                            if checkpoint:
                                checkpoint.mark_city_failed(city_name, phase, iso3, error_msg)
                            logger.info(f"Skipping {city_name} and continuing with next city...")
                            continue

                        error_msg = f"Metrics calculation failed for {city_name}: {result.get('error')}"
                        logger.error(error_msg)
                        phase_failed.append(city_name)
                        if phase == 3 and country_tracker and iso3 in country_tracker:
                            country_tracker[iso3]['failed'].append(city_name)
                        if checkpoint:
                            checkpoint.mark_city_failed(city_name, phase, iso3, error_msg)
                        email_notifier.send_error_notification(
                            error_message=error_msg, city_name=city_name, phase=phase,
                            context={'iso3': iso3, 'step': 'Metrics Calculation'}
                        )
                        if config.STRICT_MODE:
                            logger.error(f"STRICT MODE: Halting due to metrics failure in {city_name}")
                            logger.error("Fix the error and restart the pipeline to resume from checkpoint")
                            sys.exit(1)
                        continue

                    if checkpoint:
                        checkpoint.mark_city_completed(city_name, phase, iso3)
                    no_data_tracker.reset_country_consecutive_failures(iso3)
                    phase_successful.append(city_name)
                    if phase == 3 and country_tracker and iso3 in country_tracker:
                        country_tracker[iso3]['completed'].append(city_name)
                    logger.info(f"{city_name} completed successfully (Phase {phase}, {i}/{len(pending_cities)})")

                    # Strategic cache clearing to bound memory growth over a
                    # run spanning thousands of cities: every city in Phase 1
                    # (cross-country, unpredictable memory footprint), every
                    # 50 cities in Phase 2/3, and whenever a country finishes.
                    should_clear_cache = False
                    if phase == 1:
                        should_clear_cache = True
                    elif phase in [2, 3]:
                        if len(phase_successful) % 50 == 0:
                            should_clear_cache = True
                        if phase == 3 and country_tracker and iso3 in country_tracker:
                            country_data = country_tracker[iso3]
                            if len(country_data['completed']) + len(country_data['failed']) == len(country_data['total']):
                                should_clear_cache = True

                    if should_clear_cache:
                        import gc
                        clear_cache()
                        gc.collect()

                    if phase == 1 and len(phase_successful) % 10 == 0:
                        elapsed = time.time() - phase_start_time
                        avg_time_per_city = elapsed / len(phase_successful) if phase_successful else 0
                        remaining_cities = len(pending_cities) - i
                        estimated_remaining_time = avg_time_per_city * remaining_cities
                        logger.info(f"\n{'='*80}")
                        logger.info(f"PHASE 1 PROGRESS UPDATE: {len(phase_successful)}/{len(pending_cities)} cities completed")
                        logger.info(f"{'='*80}\n")
                        email_notifier.send_phase1_progress(
                            completed_count=len(phase_successful), total_count=len(pending_cities),
                            recent_cities=phase_successful[-10:], failed_cities=phase_failed,
                            elapsed_time=elapsed, estimated_remaining_time=estimated_remaining_time
                        )

                    if phase == 3 and country_tracker and iso3 in country_tracker:
                        country_data = country_tracker[iso3]
                        if len(country_data['completed']) + len(country_data['failed']) == len(country_data['total']):
                            country_time = time.time() - country_data['start_time']
                            country_name = config.get_country_name_from_iso3(iso3) or iso3
                            logger.info(f"\n{'='*80}")
                            logger.info(f"COUNTRY COMPLETED: {country_name} ({iso3})")
                            logger.info(f"  Total cities: {len(country_data['total'])}")
                            logger.info(f"  Successful: {len(country_data['completed'])}")
                            logger.info(f"  Failed: {len(country_data['failed'])}")
                            logger.info(f"  Time: {country_time/3600:.2f} hours")
                            logger.info(f"{'='*80}\n")
                            email_notifier.send_country_completion(
                                country_name=country_name, iso3=iso3, phase=phase,
                                total_cities=len(country_data['total']), successful_cities=country_data['completed'],
                                failed_cities=country_data['failed'], total_time=country_time
                            )

                except Exception as e:
                    error_msg = f"Unexpected error processing {city_name}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    phase_failed.append(city_name)
                    if phase == 3 and country_tracker and iso3 in country_tracker:
                        country_tracker[iso3]['failed'].append(city_name)
                    if checkpoint:
                        checkpoint.mark_city_failed(city_name, phase, iso3, error_msg)
                    email_notifier.send_error_notification(
                        error_message=error_msg, city_name=city_name, phase=phase,
                        context={'iso3': iso3, 'step': 'Unexpected Error', 'exception': str(e)}
                    )
                    if config.STRICT_MODE:
                        logger.error(f"STRICT MODE: Halting due to unexpected error in {city_name}")
                        logger.error("Fix the error and restart the pipeline to resume from checkpoint")
                        sys.exit(1)
                    continue

            phase_time = time.time() - phase_start_time
            logger.info("\n" + "=" * 80)
            logger.info(f"PHASE {phase} COMPLETED")
            logger.info(f"  Time: {phase_time/3600:.2f} hours")
            logger.info(f"  Successful: {len(phase_successful)}/{len(all_phase_cities)}")
            if phase_failed:
                logger.info(f"  Failed: {len(phase_failed)}")
            if checkpoint:
                stats = checkpoint.get_statistics()
                logger.info(f"  Total completed (all phases): {stats[f'phase{phase}_completed']}")
                logger.info(f"  Total failed (all phases): {stats[f'phase{phase}_failed']}")
            logger.info("=" * 80)

            if phase == 1:
                email_notifier.send_phase1_completion(
                    total_cities=len(all_phase_cities), successful_cities=phase_successful,
                    failed_cities=phase_failed, total_time=phase_time
                )
            elif phase == 2:
                email_notifier.send_phase2_completion(
                    total_cities=len(all_phase_cities), successful_cities=phase_successful,
                    failed_cities=phase_failed, total_time=phase_time
                )

        logger.info("\n" + "=" * 80)
        logger.info("ALL PHASES COMPLETED SUCCESSFULLY")
        if checkpoint:
            checkpoint.print_statistics()
            phase_stats = checkpoint.get_statistics()
            total_pipeline_time = time.time() - start_time
            email_notifier.send_final_completion(phase_stats=phase_stats, total_time=total_pipeline_time)
        logger.info("=" * 80)

    else:
        logger.error(f"Invalid PROCESSING_MODE: {config.PROCESSING_MODE}")
        logger.error("Valid modes: 'single_city', 'city_batch', 'country_batch', 'phased_automatic'")
        sys.exit(1)

    elapsed_time = time.time() - start_time

    # Regenerate the column name mapping after every run so it always
    # reflects the metric set actually computed by this codebase.
    mapping_output = config.OUTPUT_BASE / "data" / "reference" / "column_name_mapping.csv"
    generate_column_mapping_with_categories(mapping_output)
    logger.info(f"Column name mapping updated: {mapping_output}")

    if no_data_tracker.get_city_count() > 0:
        logger.info(no_data_tracker.get_summary())

    logger.info("\n" + "=" * 80 + f"\nCOMPLETED - {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        try:
            notifier = EmailNotifier(logger=logger)
            notifier.send_error_notification(
                error_message=f"Fatal error caused pipeline termination:\n\n{str(e)}",
                context={'error_type': type(e).__name__, 'traceback': 'See log file for full traceback'}
            )
        except Exception as email_error:
            logger.error(f"Failed to send error notification email: {str(email_error)}")
        sys.exit(1)
