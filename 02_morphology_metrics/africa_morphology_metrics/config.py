"""
Declarative configuration for the Africa urban morphology pipeline.

Covers:
- The phased city-processing plan (a fixed sample for Phase 1, Ghana's
  remaining cities for Phase 2, and the rest of Africa for Phase 3), plus
  the country/region groupings used to filter and report on that plan.
- The ``OpenBuildingsGCSConfig`` schema for the GCS-based Open Buildings
  extraction (see ``gcs_buildings.py``).

Cities are processed in a fixed order with no auto-skip on failure: if any
city fails, processing halts, so that a completed run is guaranteed to
cover every requested city rather than silently skipping ones that errored.
"""

from typing import Optional, Literal
from pathlib import Path
from pydantic import BaseModel, Field, field_validator

# ============================================================================
# PHASE 1: 50 SAMPLE CITIES (ALL AFRICA, INCLUDING NORTH AFRICA)
# ============================================================================
# Selection criteria:
# - All African regions: North, West, East, South, Central Africa
# - Size diversity: small (40%), medium (18%), large (26%), mega (16%)
# - Geographic diversity: 26 countries, balanced Anglophone/Francophone/Other
# - City types: mix of capitals, ordinary administrative cities, regional hubs
#
# The sample favors "ordinary" cities over capitals/tourism centers to avoid
# morphology diversity gaps, e.g. non-capital North African cities and
# non-mega Central African cities, since capitals and mega-cities alone
# under-represent typical urban form.

PHASE1_SAMPLE_CITIES = [
    # ===== SMALL CITIES FIRST (30 CITIES) =====
    # Pipeline is validated on manageable cities before tackling mega-cities.

    # First 10 kept fixed for checkpoint continuity across pipeline revisions.
    "Apam",           # Ghana - coastal fishing town, ~50K
    "Wa",             # Ghana - northern regional capital, ~100K
    "Bawku",          # Ghana - border trading town, ~80K
    "Kaolack",        # Senegal - peanut trade hub, ~200K
    "Sokode",         # Togo - central highlands, ~150K
    "Kara",           # Togo - northern commercial center, ~100K
    "Ho",             # Ghana - Volta region capital, ~100K
    "Bolgatanga",     # Ghana - Upper East capital, ~80K
    "Iringa",         # Tanzania - Southern Highlands, ~150K
    "Bukoba",         # Tanzania - Lake Victoria coffee region, ~100K

    # West Africa - Francophone ordinary cities (6)
    "Ziguinchor",     # Senegal - Casamance region, ~200K
    "Korhogo",        # Côte d'Ivoire - northern commercial, ~200K
    "Segou",          # Mali - Niger River historic town, ~130K (no accent in data)
    "Maradi",         # Niger - trading city, ~250K
    "Parakou",        # Benin - northern transport hub, ~250K
    "Kankan",         # Guinea - second city, ~200K

    # North Africa - Ordinary non-capital cities (4)
    "Sfax",           # Tunisia - industrial port, ~950K (non-tourism)
    "Oran",           # Algeria - port city, ~850K (ordinary)
    "Fez",            # Morocco - historic non-capital, ~1.1M
    "Aswan",          # Egypt - Upper Nile region, ~300K

    # East Africa - Diverse ordinary cities (4)
    "Mwanza",         # Tanzania - Lake Victoria port, ~700K
    "Mbarara",        # Uganda - western pastoral region, ~200K
    "Arusha",         # Tanzania - northern highlands, ~400K
    "Bahir Dar",      # Ethiopia - Lake Tana region, ~300K

    # Southern Africa - Small ordinary cities (3)
    "Francistown",    # Botswana - second city, mining, ~100K
    "Maun",           # Botswana - Okavango Delta gateway, ~60K
    "Livingstone",    # Zambia - Victoria Falls, ~150K

    # Central Africa - Ordinary administrative cities (3)
    "Bafoussam",      # Cameroon - western highlands, ~300K (ordinary morphology)
    "Pointe Noire",   # Congo-Brazzaville - port city, ~700K (space, not hyphen, in data)
    "Bukavu",         # DRC - Lake Kivu border, ~900K

    # ===== MAJOR CITIES (20 CITIES) =====
    # Capitals and critical regional hubs only.

    # North Africa capitals (4)
    "Cairo",          # Egypt - Mega-city (>20M), capital
    "Casablanca",     # Morocco - Economic capital, ~4M
    "Algiers",        # Algeria - Capital, ~3M
    "Tunis",          # Tunisia - Capital, ~2M

    # West Africa - Major hubs (5)
    "Lagos",          # Nigeria - Mega-city (>14M), economic hub
    "Accra",          # Ghana - Capital, tech hub, ~2.5M
    "Dakar",          # Senegal - Capital, peninsula city, ~3M
    "Abidjan",        # Côte d'Ivoire - Economic capital, ~5M
    "Ouagadougou",    # Burkina Faso - Capital, ~2.5M

    # East Africa - Major hubs (5)
    "Nairobi",        # Kenya - Regional hub, UN city, ~4.5M
    "Addis Ababa",    # Ethiopia - AU headquarters, ~5M
    "Dar es Salaam",  # Tanzania - Economic capital, ~6M
    "Kampala",        # Uganda - Capital, ~3M
    "Kigali",         # Rwanda - Clean city model, ~1M

    # Southern Africa - Major hubs (3)
    "Johannesburg",   # South Africa - Economic hub, ~5.5M
    "Durban",         # South Africa - Port city, ~3.5M
    "Lusaka",         # Zambia - Capital, ~2.5M

    # Central Africa - Mix of capitals and ordinary cities (3)
    "Kinshasa",       # DRC - Mega-city (>15M), capital
    "Luanda",         # Angola - Capital, oil rich, ~8M
    "Yaounde",        # Cameroon - Political capital, ~4M (no accent in data)
]

assert len(PHASE1_SAMPLE_CITIES) == 50, f"Expected 50 sample cities, got {len(PHASE1_SAMPLE_CITIES)}"

# ============================================================================
# PHASE 2: GHANA REMAINING CITIES
# ============================================================================
# All Ghana cities excluding those already covered in Phase 1.
PHASE2_COUNTRY_ISO3 = "GHA"
PHASE2_EXCLUDE_CITIES = [city for city in PHASE1_SAMPLE_CITIES if city in ["Accra", "Kumasi", "Tamale"]]

# ============================================================================
# PHASE 3: ALL REMAINING AFRICA
# ============================================================================
# All cities from all African countries, excluding Phase 1 and Phase 2
# cities. Computed dynamically from the agglomeration file at runtime.

# ============================================================================
# ALL AFRICA COUNTRY FILTER (57 COUNTRIES)
# ============================================================================
# ISO3 codes for all African countries, per africa_boundaries.gpkg.
ALL_AFRICA_ISO3 = [
    # North Africa (7 countries)
    "DZA",  # Algeria
    "EGY",  # Egypt
    "LBY",  # Libya
    "MAR",  # Morocco
    "SDN",  # Sudan
    "TUN",  # Tunisia
    "ESH",  # Western Sahara

    # West Africa (16 countries)
    "BEN", "BFA", "CPV", "CIV", "GMB", "GHA", "GIN", "GNB", "LBR", "MLI",
    "MRT", "NER", "NGA", "SEN", "SLE", "TGO",

    # East Africa (18 countries)
    "BDI", "COM", "DJI", "ERI", "ETH", "KEN", "MDG", "MWI", "MUS", "MOZ",
    "RWA", "SYC", "SOM", "SSD", "TZA", "UGA", "ZMB", "ZWE",

    # Southern Africa (5 countries)
    "BWA", "LSO", "NAM", "ZAF", "SWZ",

    # Central Africa (9 countries)
    "AGO", "CMR", "CAF", "TCD", "COG", "COD", "GNQ", "GAB", "STP",
]

# Kept for backward compatibility with earlier country-filter call sites.
SUB_SAHARAN_AFRICA_ISO3 = [iso3 for iso3 in ALL_AFRICA_ISO3 if iso3 not in ["DZA", "EGY", "LBY", "MAR", "SDN", "TUN", "ESH"]]

# ============================================================================
# GEOGRAPHIC REGION MAPPING (for analysis/reporting)
# ============================================================================
REGION_MAPPING = {
    # North Africa
    "DZA": "North", "EGY": "North", "LBY": "North", "MAR": "North",
    "SDN": "North", "TUN": "North", "ESH": "North",

    # West Africa
    "BEN": "West", "BFA": "West", "CPV": "West", "CIV": "West", "GMB": "West",
    "GHA": "West", "GIN": "West", "GNB": "West", "LBR": "West", "MLI": "West",
    "MRT": "West", "NER": "West", "NGA": "West", "SEN": "West", "SLE": "West",
    "TGO": "West",

    # East Africa
    "BDI": "East", "COM": "East", "DJI": "East", "ERI": "East", "ETH": "East",
    "KEN": "East", "MDG": "East", "MWI": "East", "MUS": "East", "MOZ": "East",
    "RWA": "East", "SYC": "East", "SOM": "East", "SSD": "East", "TZA": "East",
    "UGA": "East", "ZMB": "East", "ZWE": "East",

    # Southern Africa
    "BWA": "Southern", "LSO": "Southern", "NAM": "Southern", "ZAF": "Southern",
    "SWZ": "Southern",

    # Central Africa
    "AGO": "Central", "CMR": "Central", "CAF": "Central", "TCD": "Central",
    "COG": "Central", "COD": "Central", "GNQ": "Central", "GAB": "Central",
    "STP": "Central",
}


def validate_phase1_cities():
    """Validate that the Phase 1 sample meets its diversity criteria."""
    print("\n" + "="*80)
    print("VALIDATING PHASE 1 SAMPLE CITIES (ALL AFRICA)")
    print("="*80)

    print(f"\nTotal sample cities: {len(PHASE1_SAMPLE_CITIES)}")

    duplicates = [city for city in PHASE1_SAMPLE_CITIES if PHASE1_SAMPLE_CITIES.count(city) > 1]
    if duplicates:
        print(f"⚠️  WARNING: Duplicate cities found: {duplicates}")
    else:
        print("✓ No duplicates")

    north_africa_cities = [
        "Cairo", "Algiers", "Oran", "Casablanca", "Rabat",
        "Tunis", "Tripoli", "Khartoum", "Asmara"
    ]
    west_africa_cities = [
        "Lagos", "Kano", "Accra", "Kumasi", "Tamale", "Dakar",
        "Abidjan", "Bamako", "Ouagadougou", "Niamey",
        "Conakry", "Freetown", "Monrovia", "Cotonou"
    ]
    east_africa_cities = [
        "Nairobi", "Mombasa", "Dar es Salaam", "Mwanza", "Moshi", "Arusha", "Mbeya",
        "Kampala", "Addis Ababa", "Kigali", "Mogadishu", "Lilongwe", "Antananarivo"
    ]
    southern_africa_cities = [
        "Johannesburg", "Durban", "Harare", "Maputo", "Lusaka", "Gaborone", "Windhoek"
    ]
    central_africa_cities = [
        "Kinshasa", "Mbuji-Mayi", "Luanda", "Douala", "Yaounde", "N'Djamena", "Bangui"
    ]

    north_count = sum(1 for city in PHASE1_SAMPLE_CITIES if city in north_africa_cities)
    west_count = sum(1 for city in PHASE1_SAMPLE_CITIES if city in west_africa_cities)
    east_count = sum(1 for city in PHASE1_SAMPLE_CITIES if city in east_africa_cities)
    south_count = sum(1 for city in PHASE1_SAMPLE_CITIES if city in southern_africa_cities)
    central_count = sum(1 for city in PHASE1_SAMPLE_CITIES if city in central_africa_cities)

    print(f"\nRegional distribution:")
    print(f"  North Africa:    {north_count} cities ({north_count/50*100:.0f}%)")
    print(f"  West Africa:     {west_count} cities ({west_count/50*100:.0f}%)")
    print(f"  East Africa:     {east_count} cities ({east_count/50*100:.0f}%)")
    print(f"  Southern Africa: {south_count} cities ({south_count/50*100:.0f}%)")
    print(f"  Central Africa:  {central_count} cities ({central_count/50*100:.0f}%)")

    print("\n" + "="*80)


# ============================================================================
# OPEN BUILDINGS GCS EXTRACTION CONFIG
# ============================================================================

class OpenBuildingsGCSConfig(BaseModel):
    """
    Configuration for GCS-based Open Buildings extraction.

    This is the primary building-footprint extraction method used by the
    pipeline; it is significantly faster than the Earth Engine approach and
    needs no authentication, since the Open Buildings bucket is public.

    Example:
        >>> config = OpenBuildingsGCSConfig(
        ...     aoi_file=Path("study_area.geojson"),
        ...     output_dir=Path("./buildings/"),
        ...     confidence_threshold=0.75,
        ...     num_workers=4
        ... )
    """

    # ==================== Required Inputs ====================
    aoi_file: Path = Field(
        ...,
        description="Area of Interest boundary file (GeoJSON, Shapefile, etc.)"
    )

    output_dir: Path = Field(
        ...,
        description="Output directory for extracted buildings"
    )

    # ==================== Data Source Settings ====================
    data_type: Literal["polygons", "points"] = Field(
        default="polygons",
        description="Type of building data to extract (polygons include geometry)"
    )

    s2_level: int = Field(
        default=6,
        ge=4,
        le=8,
        description="S2 cell level - must match GCS bucket structure (default: 6)"
    )

    gcs_bucket_path: str = Field(
        default="gs://open-buildings-data/v3/polygons_s2_level_6_gzip_no_header",
        description="GCS path to Open Buildings data"
    )

    # ==================== Filtering Options ====================
    confidence_threshold: float = Field(
        default=0.75,
        ge=0.5,
        le=1.0,
        description="Minimum building confidence score (0.5-1.0)"
    )

    min_area_m2: float = Field(
        default=10.0,
        ge=0.0,
        description="Minimum building area in square meters"
    )

    max_area_m2: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Maximum building area in square meters (None for no limit)"
    )

    # ==================== Export Settings ====================
    export_format: Literal["geojson", "shapefile", "csv", "geoparquet"] = Field(
        default="geojson",
        description="Output file format"
    )

    include_confidence: bool = Field(
        default=True,
        description="Include confidence scores in output"
    )

    include_area: bool = Field(
        default=True,
        description="Include area calculations in output"
    )

    include_plus_codes: bool = Field(
        default=True,
        description="Include Plus Codes in output"
    )

    overwrite_existing: bool = Field(
        default=False,
        description="Overwrite existing output files"
    )

    # ==================== Performance Settings ====================
    num_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of parallel workers for downloading S2 cells"
    )

    chunk_size: int = Field(
        default=2_000_000,
        ge=100_000,
        description="Number of records to process per chunk"
    )

    # ==================== Authentication (optional for public data) ====================
    service_account_key: Optional[Path] = Field(
        default=None,
        description="Path to GCS service account key (not required for Open Buildings)"
    )

    use_anonymous_access: bool = Field(
        default=True,
        description="Use anonymous access for public data (recommended for Open Buildings)"
    )

    # ==================== Validators ====================
    @field_validator('aoi_file')
    @classmethod
    def validate_aoi_exists(cls, v: Path) -> Path:
        """Validate that the AOI file exists."""
        if not v.exists():
            raise ValueError(f"AOI file not found: {v}")
        return v

    @field_validator('service_account_key')
    @classmethod
    def validate_service_account(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate the service account key file, if provided."""
        if v is not None and not v.exists():
            raise ValueError(f"Service account key file not found: {v}")
        return v

    @field_validator('max_area_m2')
    @classmethod
    def validate_area_range(cls, v: Optional[float], info) -> Optional[float]:
        """Validate that max_area_m2 exceeds min_area_m2 when both are set."""
        if v is not None:
            min_area = info.data.get('min_area_m2', 0.0)
            if v <= min_area:
                raise ValueError(
                    f"max_area_m2 ({v}) must be greater than min_area_m2 ({min_area})"
                )
        return v

    @field_validator('gcs_bucket_path')
    @classmethod
    def validate_gcs_path(cls, v: str) -> str:
        """Validate the GCS bucket path format."""
        if not v.startswith('gs://'):
            raise ValueError(f"GCS path must start with 'gs://': {v}")
        return v

    # ==================== Helper Methods ====================
    def get_output_file_path(self) -> Path:
        """
        Generate the output file path based on the configured export format.

        Returns:
            Path object for the output file.
        """
        extensions = {
            "geojson": ".geojson",
            "shapefile": ".shp",
            "csv": ".csv",
            "geoparquet": ".parquet"
        }
        extension = extensions.get(self.export_format, ".geojson")
        return self.output_dir / f"open_buildings{extension}"

    def get_gcs_file_pattern(self) -> str:
        """
        Get the GCS file glob pattern for S2 cell files.

        Returns:
            String pattern for GCS files.
        """
        return f"{self.gcs_bucket_path}/*.csv.gz"

    def estimate_memory_usage(self) -> float:
        """
        Estimate memory usage in MB based on the configured settings.

        Returns:
            Estimated memory usage in megabytes.
        """
        # Rough estimate: chunk_size records * ~200 bytes/record.
        memory_per_worker = (self.chunk_size * 200) / (1024 * 1024)
        total_memory = memory_per_worker * self.num_workers
        return total_memory

    def summary(self) -> dict:
        """
        Get a summary of the key configuration settings.

        Returns:
            Dictionary with a configuration summary.
        """
        return {
            "aoi_file": str(self.aoi_file),
            "output_dir": str(self.output_dir),
            "output_format": self.export_format,
            "confidence_threshold": self.confidence_threshold,
            "area_filter": f"{self.min_area_m2}-{self.max_area_m2 or 'unlimited'} m²",
            "parallel_workers": self.num_workers,
            "s2_level": self.s2_level,
            "data_source": self.gcs_bucket_path,
            "estimated_memory_mb": round(self.estimate_memory_usage(), 2)
        }

    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True
        validate_assignment = True
        json_schema_extra = {
            "example": {
                "aoi_file": "study_area.geojson",
                "output_dir": "./buildings/",
                "confidence_threshold": 0.75,
                "min_area_m2": 10.0,
                "export_format": "geojson",
                "num_workers": 4
            }
        }


class OpenBuildingsGCSPointsConfig(OpenBuildingsGCSConfig):
    """
    Configuration for extracting building centroid points only.

    Faster and lighter-weight than polygon extraction, since it carries no
    geometry beyond a single point per building.
    """

    data_type: Literal["points"] = Field(
        default="points",
        description="Extract building centroids only"
    )

    gcs_bucket_path: str = Field(
        default="gs://open-buildings-data/v3/points_s2_level_6_gzip",
        description="GCS path to Open Buildings points data"
    )

    # Points carry no geometry column, so export is always tabular.
    export_format: Literal["csv", "geojson"] = Field(
        default="csv",
        description="Output format (CSV for tabular, GeoJSON to create points)"
    )


if __name__ == "__main__":
    validate_phase1_cities()
