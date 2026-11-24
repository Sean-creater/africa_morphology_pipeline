"""
OSM highway network acquisition for the building morphology pipeline.

Downloads country-level OSM road networks from Geofabrik, caches the PBF
files, and clips the network to each city's boundary. Processing is
country-first: cities are grouped by ISO3 country code, the country's PBF
is downloaded and extracted once, and each city's highways are clipped from
that shared in-memory network before it is released.

Directory layout produced:
    {base_output_dir}/{Country}/{ISO3}_highways_full.{geojson,parquet}
    {base_output_dir}/{Country}/{City}/highways/{City}_highways.geojson
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import gc
import time
import geopandas as gpd
import pandas as pd
from multiprocessing import cpu_count

from .global_data_cache import get_agglomeration_data, get_boundaries_data
from .projection import get_crs_for_city, get_crs_for_country
from .geofabrik import get_cached_pbf
from .osm_clipping import clip_highways_to_aoi

logger = logging.getLogger(__name__)



class IntegratedOSMHighwaysProcessor:
    """
    OSM highways processor for the building analysis pipeline.

    Workflow:
    1. Group cities by country.
    2. For each country:
       a. Download the country-level OSM PBF (once per country, cached).
       b. Extract the full highway network for that country.
       c. Clip the network to each city boundary and save.
    3. Return a mapping of city name to highway file path.

    Grouping by country minimizes redundant PBF downloads and lets the
    (expensive) country-level network extraction be amortized across all
    cities in that country.
    """

    def __init__(
        self,
        base_output_dir: Path,
        agglomeration_file: Path,
        grid_data_base: Path,
        country_boundaries_file: Path,
        pbf_cache_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
        force_redownload: bool = False,
        overwrite_existing: bool = False,
        num_workers: int = None
    ):
        """
        Initialize the OSM highways processor.

        Args:
            base_output_dir: Base directory for outputs (e.g., data/{Country}/...).
            agglomeration_file: Path to agglomerations.gpkg.
            grid_data_base: Path to the grid boundaries base directory.
            country_boundaries_file: Path to the country boundaries file used
                to resolve ISO3 codes to country names (see README for the
                expected schema).
            pbf_cache_dir: Directory to cache downloaded PBF files
                (default: ~/.geoworkflow/osm_cache).
            logger: Logger instance.
            force_redownload: Force re-download of PBF files even if cached.
            overwrite_existing: Overwrite existing clipped highway files.
            num_workers: Number of parallel workers (default: cpu_count-derived).
        """
        self.base_output_dir = Path(base_output_dir)
        self.agglomeration_file = Path(agglomeration_file)
        self.grid_data_base = Path(grid_data_base)
        self.country_boundaries_file = Path(country_boundaries_file)
        self.pbf_cache_dir = pbf_cache_dir or (Path.home() / ".geoworkflow" / "osm_cache")
        self.logger = logger or logging.getLogger(__name__)
        self.force_redownload = force_redownload
        self.overwrite_existing = overwrite_existing
        self.num_workers = num_workers or min(16, max(4, cpu_count()))

        # ISO3 to country name mapping (for directory names)
        self.iso3_to_country = self._build_iso3_to_country_mapping()

        # Cache for country-level highway networks (cleared after each country)
        self._country_highways_cache: Dict[str, gpd.GeoDataFrame] = {}

        # ISO3 to Geofabrik region mapping
        self.iso3_to_geofabrik = self._build_iso3_to_geofabrik_mapping()

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """
        Sanitize a city/country name for use in file paths.

        Replaces path separators and characters invalid on common filesystems
        (``/ \\ : ? * | " < >``) so that names can be used directly as
        directory and file components.
        """
        name = name.replace('/', '_').replace('\\', '_')
        name = name.replace(':', '_')
        for char in ['?', '*', '|', '"', '<', '>']:
            name = name.replace(char, '')
        name = name.strip().strip('_')
        return name

    def _build_iso3_to_country_mapping(self) -> Dict[str, str]:
        """
        Build a mapping from ISO3 codes to standardized country names, used
        for directory naming. Backed by the shared boundaries cache so the
        source file is read once per process regardless of how many
        processors/batches are constructed.
        """
        try:
            if not self.agglomeration_file.exists():
                self.logger.warning(f"Agglomeration file not found: {self.agglomeration_file}")
                return {}

            if not self.country_boundaries_file.exists():
                self.logger.error(f"Country boundaries file not found: {self.country_boundaries_file}")
                return {}

            countries = get_boundaries_data(self.country_boundaries_file)
            if countries is None:
                return {}

            mapping = {}
            for _, row in countries.iterrows():
                iso3 = row.get("ISO")
                country_name = row.get("NAME_0", "")
                if iso3 and country_name:
                    # Keep spaces (grid workflow uses original names); only
                    # strip characters that are actually invalid in paths.
                    clean_name = country_name.replace("'", "").replace("(", "").replace(")", "")
                    mapping[iso3] = clean_name
            return mapping

        except Exception as e:
            self.logger.error(f"Error building ISO3 mapping: {str(e)}")
            return {}

    def _build_iso3_to_geofabrik_mapping(self) -> Dict[str, str]:
        """
        Build a mapping from ISO3 codes to Geofabrik region names.

        Geofabrik uses lowercase country names with hyphens for multi-word
        names, e.g. "Egypt" -> "egypt", "South Africa" -> "south-africa".
        Source: https://download.geofabrik.de/africa/

        Returns:
            Dictionary mapping ISO3 -> Geofabrik region name.
        """
        mapping = {
            # North Africa
            "DZA": "algeria",
            "EGY": "egypt",
            "LBY": "libya",
            "MAR": "morocco",
            "SDN": "sudan",
            "TUN": "tunisia",

            # West Africa
            "BEN": "benin",
            "BFA": "burkina-faso",
            "CPV": "cape-verde",
            "CIV": "ivory-coast",
            "GMB": "senegal-and-gambia",  # Gambia is combined with Senegal on Geofabrik
            "GHA": "ghana",
            "GIN": "guinea",
            "GNB": "guinea-bissau",
            "LBR": "liberia",
            "MLI": "mali",
            "MRT": "mauritania",
            "NER": "niger",
            "NGA": "nigeria",
            "SEN": "senegal-and-gambia",  # Senegal is combined with Gambia on Geofabrik
            "SLE": "sierra-leone",
            "TGO": "togo",

            # East Africa
            "BDI": "burundi",
            "COM": "comores",  # Geofabrik uses "comores", not "comoros"
            "DJI": "djibouti",
            "ERI": "eritrea",
            "ETH": "ethiopia",
            "KEN": "kenya",
            "MDG": "madagascar",
            "MWI": "malawi",
            "MUS": "mauritius",
            "MOZ": "mozambique",
            "RWA": "rwanda",
            "SYC": "seychelles",
            "SOM": "somalia",
            "SSD": "south-sudan",
            "TZA": "tanzania",
            "UGA": "uganda",
            "ZMB": "zambia",
            "ZWE": "zimbabwe",

            # Southern Africa
            "BWA": "botswana",
            "LSO": "lesotho",  # Lesotho is separate on Geofabrik
            "NAM": "namibia",
            "ZAF": "south-africa",
            "SWZ": "swaziland",  # Geofabrik still uses "swaziland", not "eswatini"

            # Central Africa
            "AGO": "angola",
            "CMR": "cameroon",
            "CAF": "central-african-republic",
            "TCD": "chad",
            "COG": "congo-brazzaville",
            "COD": "congo-democratic-republic",
            "GNQ": "equatorial-guinea",
            "GAB": "gabon",
            "STP": "sao-tome-and-principe",
        }

        return mapping

    def process_cities_batch(self, city_names: List[str]) -> Dict[str, Optional[Path]]:
        """
        Process multiple cities in batch mode.

        Main entry point for the building analysis pipeline. Groups cities
        by country and processes each country's cities against a single
        shared download/extraction of that country's highway network.

        Args:
            city_names: List of city names to process.

        Returns:
            Dictionary mapping city_name -> highway_file_path (or None if failed).
        """
        self.logger.info(f"Processing OSM highways for {len(city_names)} cities")

        cities_by_country = self._group_cities_by_country(city_names)

        if not cities_by_country:
            self.logger.error("No valid cities found to process")
            return {city: None for city in city_names}

        results = {}
        for iso3, cities in cities_by_country.items():
            country_name = self.iso3_to_country.get(iso3, iso3)
            self.logger.info(f"\nProcessing {len(cities)} cities in {country_name} ({iso3})")

            country_highways = self._get_country_highways(iso3, country_name)

            if country_highways is None:
                self.logger.error(f"Failed to download highways for {country_name}")
                for city in cities:
                    results[city['name']] = None
                continue

            for city in cities:
                city_result = self._process_single_city(
                    city_name=city['name'],
                    city_boundary=city['boundary'],
                    country_name=country_name,
                    country_highways=country_highways,
                    iso3=iso3
                )
                results[city['name']] = city_result

            # Release the country-level network from memory before moving to
            # the next country, to bound peak memory across a large batch.
            if iso3 in self._country_highways_cache:
                self.logger.info(f"  Clearing memory cache for {country_name} (freeing memory)")
                del self._country_highways_cache[iso3]
                gc.collect()

        successful = sum(1 for v in results.values() if v is not None)
        self.logger.info(f"\n✓ OSM highways processed: {successful}/{len(city_names)} cities")

        return results

    def _group_cities_by_country(self, city_names: List[str]) -> Dict[str, List[Dict]]:
        """
        Group cities by their ISO3 country code.

        Only processes cities present in ``city_names`` (not every city in
        the matched countries), so highways are clipped for the requested
        cities alone.

        Args:
            city_names: List of target city names to process.

        Returns:
            Dict mapping ISO3 code -> list of city info dicts, each
            containing ``{name, iso3, boundary}``.
        """
        try:
            agglo = get_agglomeration_data(self.agglomeration_file)
            if agglo is None:
                self.logger.error("Failed to load agglomeration data")
                return {}

            # Some city names in agglomerations.gpkg carry trailing spaces
            # (e.g. 'Armagentolis ' in SSD vs 'Armagentolis' in SDN), which
            # would otherwise break exact-name matching. Match on a
            # stripped copy without altering the source data.
            agglo['Agglomeration_Name_Clean'] = agglo['Agglomeration_Name'].str.strip()

            cities_by_country = {}
            for city_name in city_names:
                city_name_clean = city_name.strip()

                city_row = agglo[agglo['Agglomeration_Name_Clean'] == city_name_clean]

                if len(city_row) == 0:
                    self.logger.warning(f"City not found in agglomerations: {city_name_clean}")
                    continue

                # Some cities share an identical name across different
                # countries (e.g. Dondo in AGO/COD/MOZ). If multiple rows
                # match, check whether they are the same place (duplicate
                # records) or true homonyms in different locations, and
                # prefer the first match either way, with a warning.
                if len(city_row) > 1:
                    self.logger.warning(f"  ⚠️  Multiple cities named '{city_name_clean}' found ({len(city_row)} entries)")
                    self.logger.warning(f"      Countries: {city_row['ISO3'].unique()}")
                    self.logger.warning(f"      IDs: {city_row['Agglomeration_ID'].tolist()}")

                    geoms_wgs84 = gpd.GeoSeries(city_row.geometry.values, crs=agglo.crs).to_crs('EPSG:4326')
                    centroids = [g.centroid for g in geoms_wgs84]

                    if len(centroids) >= 2:
                        import numpy as np
                        c1, c2 = centroids[0], centroids[1]
                        lat_diff = abs(c1.y - c2.y)
                        lon_diff = abs(c1.x - c2.x)
                        dist_km = np.sqrt((lat_diff * 111)**2 + (lon_diff * 111 * np.cos(np.radians(c1.y)))**2)

                        if dist_km < 1:
                            self.logger.info(f"      -> Same location ({dist_km:.2f} km apart) - treating as duplicates")
                            self.logger.info(f"      -> Using first entry (ID={city_row.iloc[0]['Agglomeration_ID']})")
                        else:
                            self.logger.warning(f"      -> Different locations ({dist_km:.0f} km apart) - true homonyms!")
                            self.logger.warning(f"      -> Using first entry (ID={city_row.iloc[0]['Agglomeration_ID']}, {city_row.iloc[0]['ISO3']})")
                            self.logger.warning(f"      -> NOTE: This may not be the intended city!")

                iso3 = city_row.iloc[0]['ISO3']

                # Prefer the grid boundary (more accurate, lower complexity);
                # fall back to the AfricaPolis agglomeration boundary if the
                # grid boundary hasn't been generated for this city yet.
                country_name = self.iso3_to_country.get(iso3, iso3)
                boundary_file = self.grid_data_base / country_name / city_name_clean / f"{city_name_clean}_grid_boundary.geojson"

                if boundary_file.exists():
                    boundary = gpd.read_file(boundary_file)
                    self.logger.info(f"  Using grid boundary for {city_name_clean}")
                else:
                    self.logger.warning(f"  Grid boundary not found for {city_name_clean}, using AfricaPolis boundary")

                    boundary = gpd.GeoDataFrame(
                        {'geometry': [city_row.iloc[0].geometry]},
                        crs=agglo.crs
                    )

                    if boundary.crs != 'EPSG:4326':
                        boundary = boundary.to_crs('EPSG:4326')

                if iso3 not in cities_by_country:
                    cities_by_country[iso3] = []

                cities_by_country[iso3].append({
                    'name': city_name_clean,
                    'iso3': iso3,
                    'boundary': boundary
                })

            return cities_by_country

        except Exception as e:
            self.logger.error(f"Error grouping cities by country: {str(e)}")
            return {}

    def _get_country_highways(self, iso3: str, country_name: str) -> Optional[gpd.GeoDataFrame]:
        """
        Download and cache the country-level highway network.

        Checks, in order: the in-memory cache, an on-disk GeoParquet cache
        (fast path), an on-disk GeoJSON cache (compatible but slower), and
        finally downloads from Geofabrik and extracts via PBF.

        Args:
            iso3: ISO3 country code.
            country_name: Standardized country name.

        Returns:
            GeoDataFrame with country highways, or None if extraction failed.
        """
        if iso3 in self._country_highways_cache:
            self.logger.info(f"  Using cached highways for {country_name}")
            return self._country_highways_cache[iso3]

        country_dir = self.base_output_dir / country_name
        country_dir.mkdir(parents=True, exist_ok=True)
        country_highways_file = country_dir / f"{iso3}_highways_full.geojson"

        # GeoParquet loads 15-30x faster than GeoJSON for these
        # multi-hundred-MB to multi-GB country networks, so it is
        # preferred whenever available.
        country_parquet_file = country_dir / f"{iso3}_highways_full.parquet"

        if country_parquet_file.exists() and not self.overwrite_existing:
            self.logger.info(f"  ✓ Found Parquet file: {country_parquet_file.name}")
            try:
                file_size_mb = country_parquet_file.stat().st_size / (1024**2)
                self.logger.info(f"  Loading {file_size_mb:.0f} MB Parquet (fast: ~10-15 seconds)...")
                start_time = time.time()
                highways = gpd.read_parquet(country_parquet_file)
                load_time = time.time() - start_time
                self.logger.info(f"  ✓ Loaded {len(highways):,} highway segments in {load_time:.1f}s")
                self._country_highways_cache[iso3] = highways
                return highways
            except Exception as e:
                self.logger.warning(f"  Failed to load Parquet: {str(e)}, trying GeoJSON...")

        if country_highways_file.exists() and not self.overwrite_existing:
            self.logger.info(f"  Loading existing country highways: {country_highways_file.name}")
            try:
                file_size_gb = country_highways_file.stat().st_size / (1024**3)
                self.logger.info(f"  File size: {file_size_gb:.2f} GB - this may take 1-2 minutes to load...")
                highways = gpd.read_file(country_highways_file)
                self.logger.info(f"  ✓ Loaded {len(highways):,} highway segments")
                self._country_highways_cache[iso3] = highways

                # Opportunistically materialize the Parquet copy so later
                # runs hit the fast path above.
                if not country_parquet_file.exists():
                    self.logger.info(f"  Creating Parquet version for future speedup...")
                    try:
                        highways.to_parquet(country_parquet_file)
                        parquet_size = country_parquet_file.stat().st_size / (1024**2)
                        self.logger.info(f"  ✓ Parquet saved: {parquet_size:.0f} MB (will be used next time)")
                    except Exception as e:
                        self.logger.warning(f"  Failed to create Parquet: {str(e)}")

                return highways
            except Exception as e:
                self.logger.warning(f"  Failed to load cached file: {str(e)}, re-downloading...")

        self.logger.info(f"  Downloading OSM highways for {country_name}...")

        try:
            # Extract the whole country's network from a single PBF rather
            # than downloading per-city from AfricaPolis boundaries.
            if not self.country_boundaries_file.exists():
                self.logger.error(f"  Country boundaries file not found: {self.country_boundaries_file}")
                return None

            country_boundaries = get_boundaries_data(self.country_boundaries_file)
            if country_boundaries is None:
                self.logger.error(f"  Failed to load country boundaries")
                return None

            country_boundary = country_boundaries[country_boundaries['ISO'] == iso3]

            if len(country_boundary) == 0:
                self.logger.error(f"  Country boundary not found for {iso3}")
                return None

            temp_output_dir = country_dir / "temp_osm_download"
            temp_output_dir.mkdir(exist_ok=True)
            temp_boundary_file = temp_output_dir / f"{iso3}_boundary.geojson"
            country_boundary.to_file(temp_boundary_file, driver='GeoJSON')

            geofabrik_region = self.iso3_to_geofabrik.get(iso3)
            if geofabrik_region is None:
                self.logger.error(f"  No Geofabrik region mapping found for {iso3} ({country_name})")
                return None

            # SMART BBOX STRATEGY: large countries (e.g. Tanzania 668MB,
            # Egypt 1.8GB) can overflow memory if the full PBF is loaded
            # unbounded, so extraction is always scoped to a bounding box
            # around the target cities rather than the whole country
            # geometry, which is cheap for small countries and essential
            # for large ones.
            self.logger.info(f"  Extracting highways from PBF (smart bbox strategy)...")

            self.logger.info(f"  Loading PBF for {iso3}...")
            pbf_path, metadata = get_cached_pbf(
                region=geofabrik_region,
                cache_dir=self.pbf_cache_dir,
                force_redownload=self.force_redownload,
                max_age_days=30
            )

            if pbf_path is None:
                self.logger.error(f"  Failed to download PBF for {geofabrik_region}")
                return None

            pbf_size_mb = pbf_path.stat().st_size / (1024**2)
            self.logger.info(f"  Initializing pyrosm for {geofabrik_region} ({pbf_size_mb:.1f} MB)")

            # Extraction uses pyogrio (GDAL) reading the PBF with a bbox
            # filter first: pyrosm loading a full country PBF (e.g.
            # Tanzania 668MB) takes 8+ minutes and 24GB RAM, whereas
            # pyogrio with a bbox filter completes in 2-5 seconds using
            # 100-500MB RAM. pyrosm is kept as a fallback for PBFs or
            # queries pyogrio cannot handle.
            temp_boundary_gdf = gpd.read_file(temp_boundary_file)
            temp_boundary_wgs84 = temp_boundary_gdf.to_crs('EPSG:4326')
            bounds = temp_boundary_wgs84.total_bounds  # [minx, miny, maxx, maxy]

            # Buffer (~10km at the equator) to catch roads at the boundary edge.
            bbox = (
                bounds[0] - 0.1,
                bounds[1] - 0.1,
                bounds[2] + 0.1,
                bounds[3] + 0.1
            )

            self.logger.info(f"  Using GDAL for fast extraction with bbox: [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]")

            try:
                # pyogrio is 10-50x faster than pyrosm for this bbox-filtered read.
                import pyogrio

                highways = pyogrio.read_dataframe(
                    str(pbf_path),
                    layer='lines',
                    bbox=bbox,
                    where="highway IS NOT NULL",
                    use_arrow=True
                )

                if highways is None or len(highways) == 0:
                    self.logger.warning(f"  No highways found with pyogrio, trying pyrosm fallback...")
                    raise ValueError("No highways extracted with pyogrio")

                self.logger.info(f"  ✓ Loaded {len(highways):,} highway segments from {geofabrik_region} (GDAL)")

            except Exception as e:
                self.logger.warning(f"  pyogrio failed ({e}), using pyrosm fallback (slower)...")
                import pyrosm

                osm = pyrosm.OSM(str(pbf_path), bounding_box=bbox)
                highways = osm.get_network(network_type="all")

                if highways is None or len(highways) == 0:
                    self.logger.error(f"  No highways extracted from {geofabrik_region} PBF")
                    return None

                self.logger.info(f"  ✓ Loaded {len(highways):,} highway segments from {geofabrik_region} (pyrosm)")

            # Store the country-level network in the country's own optimal
            # CRS; cities reproject to their own CRS when clipped.
            try:
                COUNTRY_CRS = get_crs_for_country(iso3)
                self.logger.info(f"  Using country-specific CRS for {country_name} ({iso3}): {COUNTRY_CRS}")
            except Exception as e:
                self.logger.warning(f"  Failed to get country CRS: {e}, using ESRI:102022")
                COUNTRY_CRS = "ESRI:102022"

            if highways.crs != COUNTRY_CRS:
                highways = highways.to_crs(COUNTRY_CRS)

            # Persist both formats: GeoJSON for compatibility with other GIS
            # tools, and GeoParquet for the 15-30x faster reload used by
            # subsequent runs.
            self.logger.info(f"  Saving country highways in dual formats...")

            self.logger.info(f"  - GeoJSON: {country_highways_file.name}")
            highways.to_file(country_highways_file, driver='GeoJSON')

            country_parquet_file = country_highways_file.with_suffix('.parquet')
            self.logger.info(f"  - Parquet: {country_parquet_file.name} (for fast loading)")
            highways.to_parquet(country_parquet_file)

            self.logger.info(f"  ✓ Saved both formats (Parquet will be used for future loads)")

            if temp_boundary_file.exists():
                temp_boundary_file.unlink()
            if temp_output_dir.exists() and not list(temp_output_dir.iterdir()):
                temp_output_dir.rmdir()

            self._country_highways_cache[iso3] = highways

            self.logger.info(f"  ✓ Downloaded {len(highways):,} highway segments for {country_name}")
            return highways

        except Exception as e:
            self.logger.error(f"  Error downloading highways: {str(e)}")
            return None

    def _process_single_city(
        self,
        city_name: str,
        city_boundary: gpd.GeoDataFrame,
        country_name: str,
        country_highways: gpd.GeoDataFrame,
        iso3: str
    ) -> Optional[Path]:
        """
        Clip the country highway network to a single city boundary and save it.

        Args:
            city_name: City name.
            city_boundary: City boundary GeoDataFrame (from grid_boundary.geojson).
            country_name: Country name, for directory structure.
            country_highways: Full country highway network.
            iso3: ISO3 country code, for the city-specific CRS lookup.

        Returns:
            Path to the clipped highway file, or None if clipping produced
            no output.
        """
        safe_city_name = self._sanitize_filename(city_name)

        city_output_dir = self.base_output_dir / country_name / safe_city_name / "highways"
        city_output_dir.mkdir(parents=True, exist_ok=True)

        output_file = city_output_dir / f"{safe_city_name}_highways.geojson"

        if output_file.exists() and not self.overwrite_existing:
            self.logger.info(f"  ✓ {city_name}: Using existing highways")
            return output_file

        try:
            try:
                TARGET_CRS = get_crs_for_city(city_name, iso3)
                self.logger.info(f"  Using city-specific CRS for {city_name}: {TARGET_CRS}")
            except Exception as e:
                self.logger.warning(f"  Failed to get dynamic CRS for {city_name}: {e}, using ESRI:102022")
                TARGET_CRS = "ESRI:102022"

            if city_boundary.crs != TARGET_CRS:
                city_boundary = city_boundary.to_crs(TARGET_CRS)
            if country_highways.crs != TARGET_CRS:
                country_highways = country_highways.to_crs(TARGET_CRS)

            # Two-stage clipping: a fast bbox pre-filter using the spatial
            # index narrows a country network of ~millions of segments down
            # to the thousands relevant to this city, before the more
            # expensive precise geometry clip runs on that small subset.
            self.logger.info(f"  Clipping highways for {city_name}...")

            buffer_meters = 500
            city_geom = city_boundary.union_all()

            from shapely.geometry import box
            minx, miny, maxx, maxy = city_geom.bounds
            bbox_buffer = 1000  # 1km buffer to catch edge cases
            bbox_geom = box(minx - bbox_buffer, miny - bbox_buffer,
                           maxx + bbox_buffer, maxy + bbox_buffer)

            self.logger.info(f"  Stage 1: Bbox pre-filter on {len(country_highways):,} highways...")
            bbox_matches_idx = list(country_highways.sindex.query(bbox_geom, predicate='intersects'))
            bbox_filtered = country_highways.iloc[bbox_matches_idx]
            self.logger.info(f"  Bbox filter: {len(country_highways):,} → {len(bbox_filtered):,} highways ({len(bbox_filtered)/len(country_highways)*100:.1f}%)")

            self.logger.info(f"  Stage 2: Precise clipping {len(bbox_filtered):,} highways to city boundary...")

            if len(bbox_filtered) == 0:
                self.logger.warning(f"  ✗ {city_name}: No highways in city bbox (city outside OSM coverage area)")
                self.logger.info(f"     This is normal for remote areas without road infrastructure")
                return None

            clipped = clip_highways_to_aoi(
                bbox_filtered,  # Pre-filtered subset, not the full country network
                city_boundary,
                buffer_meters=buffer_meters
            )

            if len(clipped) == 0:
                self.logger.warning(f"  ✗ {city_name}: No highways found after clipping to boundary")
                return None

            # Remove duplicate and overlapping geometries (e.g. from
            # overlapping Geofabrik extracts) to avoid COINS warnings
            # downstream in network analysis.
            initial_count = len(clipped)

            clipped = clipped.drop_duplicates(subset=['geometry'], keep='first')

            if 'geometry' in clipped.columns:
                def get_line_key(geom):
                    """Hashable key for a line geometry, invariant to direction."""
                    if geom.geom_type == 'LineString':
                        coords = list(geom.coords)
                        return tuple(sorted([coords[0], coords[-1]]))
                    elif geom.geom_type == 'MultiLineString':
                        coords = list(geom.geoms[0].coords)
                        return tuple(sorted([coords[0], coords[-1]]))
                    return None

                clipped['_line_key'] = clipped.geometry.apply(get_line_key)
                clipped = clipped.drop_duplicates(subset=['_line_key'], keep='first')
                clipped = clipped.drop(columns=['_line_key'])

            duplicates_removed = initial_count - len(clipped)

            if duplicates_removed > 0:
                self.logger.info(f"  Removed {duplicates_removed} duplicate/overlapping highway segments")

            clipped.to_file(output_file, driver='GeoJSON')

            self.logger.info(f"  ✓ {city_name}: Saved {len(clipped):,} highway segments -> {output_file.name}")
            return output_file

        except Exception as e:
            self.logger.error(f"  ✗ {city_name}: Error clipping highways: {str(e)}")
            return None


def process_osm_highways_for_cities(
    city_names: List[str],
    base_output_dir: Path,
    agglomeration_file: Path,
    grid_data_base: Path,
    country_boundaries_file: Path,
    logger: Optional[logging.Logger] = None,
    force_redownload: bool = False,
    overwrite_existing: bool = False
) -> Dict[str, Optional[Path]]:
    """
    Process OSM highways for a list of cities.

    Main entry point called by the pipeline orchestrator.

    Args:
        city_names: List of city names to process.
        base_output_dir: Base output directory (e.g., data/).
        agglomeration_file: Path to agglomerations.gpkg.
        grid_data_base: Path to the grid data base directory.
        country_boundaries_file: Path to the country boundaries file used to
            resolve ISO3 codes to country names.
        logger: Logger instance.
        force_redownload: Force re-download of PBF files.
        overwrite_existing: Overwrite existing city highway files.

    Returns:
        Dictionary mapping city_name -> highway_file_path (or None if failed).

    Example:
        >>> results = process_osm_highways_for_cities(
        ...     city_names=['Accra', 'Kumasi', 'Tamale'],
        ...     base_output_dir=Path('data'),
        ...     agglomeration_file=Path('/data/agglomerations.gpkg'),
        ...     grid_data_base=Path('/grid_data'),
        ...     country_boundaries_file=Path('/data/africa_boundaries.gpkg'),
        ...     logger=logger
        ... )
        >>> print(results['Accra'])  # Path to Accra_highways.geojson
    """
    processor = IntegratedOSMHighwaysProcessor(
        base_output_dir=base_output_dir,
        agglomeration_file=agglomeration_file,
        grid_data_base=grid_data_base,
        country_boundaries_file=country_boundaries_file,
        logger=logger,
        force_redownload=force_redownload,
        overwrite_existing=overwrite_existing
    )

    return processor.process_cities_batch(city_names)


# ============================================================================
# CLI for standalone testing
# ============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Download and clip OSM highways for cities")
    parser.add_argument('--cities', nargs='+', required=True, help='City names to process')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / "data",
                       help='Base output directory')
    parser.add_argument('--agglomeration-file', type=Path, required=True,
                       help='Path to the city/agglomeration boundaries file (see README for schema)')
    parser.add_argument('--grid-data-base', type=Path, required=True,
                       help='Path to the grid data base directory (see the companion '
                            'africa_hex_grid_workflow repository)')
    parser.add_argument('--country-boundaries-file', type=Path, required=True,
                       help='Path to the country boundaries file (see README for schema)')
    parser.add_argument('--force-redownload', action='store_true',
                       help='Force re-download of country PBF files')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing city highway files')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("OSM HIGHWAYS PROCESSOR - STANDALONE MODE")
    logger.info("=" * 80)
    logger.info(f"Cities: {args.cities}")
    logger.info(f"Output: {args.output_dir}")

    results = process_osm_highways_for_cities(
        city_names=args.cities,
        base_output_dir=args.output_dir,
        agglomeration_file=args.agglomeration_file,
        grid_data_base=args.grid_data_base,
        country_boundaries_file=args.country_boundaries_file,
        logger=logger,
        force_redownload=args.force_redownload,
        overwrite_existing=args.overwrite
    )

    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)
    for city, path in results.items():
        if path:
            logger.info(f"✓ {city}: {path}")
        else:
            logger.info(f"✗ {city}: Failed")

    successful = sum(1 for v in results.values() if v is not None)
    logger.info(f"\nSuccess rate: {successful}/{len(results)}")
