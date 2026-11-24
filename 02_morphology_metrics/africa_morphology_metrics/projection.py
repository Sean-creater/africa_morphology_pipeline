"""
Projection (CRS) lookup for African cities and countries.

Provides the optimal UTM coordinate reference system for any city or
country in the pipeline, backed by a pre-computed mapping
(``data/reference/projection_mapping.json``) built from each country's
centroid, with per-city overrides for cases that straddle UTM zone
boundaries or otherwise need a specific zone.
"""

import json
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Package layout: africa_morphology_metrics/africa_morphology_metrics/projection.py
# Data layout:    africa_morphology_metrics/data/reference/projection_mapping.json
DEFAULT_MAPPING_FILE = Path(__file__).parent.parent / "data" / "reference" / "projection_mapping.json"


class ProjectionMapper:
    """
    Retrieves the optimal projection CRS for African cities and countries.

    Usage:
        mapper = ProjectionMapper()
        crs = mapper.get_crs_for_city("Lagos", "NGA")
        # Returns: "EPSG:32631"
    """

    def __init__(self, mapping_file: Optional[Path] = None):
        """
        Initialize the projection mapper.

        Args:
            mapping_file: Path to projection_mapping.json. Defaults to the
                packaged reference copy under ``data/reference/``.
        """
        if mapping_file is None:
            mapping_file = DEFAULT_MAPPING_FILE

        self.mapping_file = Path(mapping_file)
        self.mapping = self._load_mapping()

    def _load_mapping(self) -> Dict:
        """Load the projection mapping from its JSON file."""
        if not self.mapping_file.exists():
            raise FileNotFoundError(
                f"Projection mapping file not found: {self.mapping_file}\n"
                f"Please run projection_mapping_generator.py first."
            )

        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        logger.info(f"✓ Loaded projection mapping: {mapping['metadata']['total_countries']} countries, "
                   f"{mapping['metadata']['cities_with_overrides']} city overrides")

        return mapping

    def get_crs_for_city(self, city_name: str, iso3: str) -> str:
        """
        Get the optimal CRS for a specific city.

        Checks for a city-specific override first (used for cities that
        straddle UTM zone boundaries or otherwise need a specific zone),
        then falls back to the country default.

        Args:
            city_name: City name (whitespace-stripped before lookup).
            iso3: Country ISO3 code.

        Returns:
            EPSG code as a string (e.g., "EPSG:32632").

        Raises:
            ValueError: If the country is not found in the mapping.
        """
        city_name_clean = city_name.strip()

        if city_name_clean in self.mapping['city_overrides']:
            override = self.mapping['city_overrides'][city_name_clean]
            logger.debug(f"Using city-specific CRS for {city_name_clean}: {override['crs']} "
                        f"(reason: {override['reason']})")
            return override['crs']

        if iso3 not in self.mapping['country_defaults']:
            raise ValueError(
                f"Country {iso3} not found in projection mapping. "
                f"Available countries: {sorted(self.mapping['country_defaults'].keys())}"
            )

        country_info = self.mapping['country_defaults'][iso3]
        logger.debug(f"Using country default CRS for {city_name_clean} ({iso3}): {country_info['crs']}")
        return country_info['crs']

    def get_crs_for_country(self, iso3: str) -> str:
        """
        Get the default CRS for a country.

        Args:
            iso3: Country ISO3 code.

        Returns:
            EPSG code as a string.

        Raises:
            ValueError: If the country is not found.
        """
        if iso3 not in self.mapping['country_defaults']:
            raise ValueError(f"Country {iso3} not found in projection mapping")

        return self.mapping['country_defaults'][iso3]['crs']

    def get_crs_info(self, city_name: Optional[str] = None, iso3: Optional[str] = None) -> Dict:
        """
        Get detailed CRS information, including metadata.

        Args:
            city_name: Optional city name for city-specific info.
            iso3: Country ISO3 code.

        Returns:
            Dictionary with 'crs', 'name', 'reason', and related metadata.
        """
        if city_name:
            city_name_clean = city_name.strip()
            if city_name_clean in self.mapping['city_overrides']:
                override = self.mapping['city_overrides'][city_name_clean]
                return {
                    'crs': override['crs'],
                    'name': f"UTM zone {override['crs'].split(':')[1][3:]}",
                    'reason': override['reason'],
                    'type': 'city_override',
                    'country_default': override['country_default']
                }

        if iso3:
            if iso3 not in self.mapping['country_defaults']:
                raise ValueError(f"Country {iso3} not found")

            country_info = self.mapping['country_defaults'][iso3].copy()
            country_info['type'] = 'country_default'
            return country_info

        raise ValueError("Must provide either city_name or iso3")

    def has_city_override(self, city_name: str) -> bool:
        """Check whether a city has a specific projection override."""
        return city_name.strip() in self.mapping['city_overrides']

    def get_countries(self) -> list:
        """Get the list of all countries in the mapping."""
        return sorted(self.mapping['country_defaults'].keys())

    def get_metadata(self) -> Dict:
        """Get the mapping's metadata block."""
        return self.mapping['metadata']


# Global singleton instance, lazily constructed on first use.
_global_mapper: Optional[ProjectionMapper] = None


def get_projection_mapper(mapping_file: Optional[Path] = None) -> ProjectionMapper:
    """
    Get the process-wide ProjectionMapper instance (singleton pattern).

    Args:
        mapping_file: Optional path to projection_mapping.json. Only used
            on the first call; later calls reuse the existing instance.

    Returns:
        ProjectionMapper instance.
    """
    global _global_mapper

    if _global_mapper is None:
        _global_mapper = ProjectionMapper(mapping_file)

    return _global_mapper


def get_crs_for_city(city_name: str, iso3: str) -> str:
    """
    Get the CRS for a city.

    Args:
        city_name: City name.
        iso3: Country ISO3 code.

    Returns:
        EPSG code as a string (e.g., "EPSG:32632").
    """
    mapper = get_projection_mapper()
    return mapper.get_crs_for_city(city_name, iso3)


def get_crs_for_country(iso3: str) -> str:
    """
    Get the default CRS for a country.

    Args:
        iso3: Country ISO3 code.

    Returns:
        EPSG code as a string.
    """
    mapper = get_projection_mapper()
    return mapper.get_crs_for_country(iso3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    mapper = ProjectionMapper()

    print("\n" + "=" * 80)
    print("PROJECTION MAPPER TEST")
    print("=" * 80)

    test_cities = [
        ("Lagos", "NGA"),
        ("Cairo", "EGY"),
        ("Johannesburg", "ZAF"),
        ("Nairobi", "KEN"),
        ("Casablanca", "MAR"),
        ("Addis Ababa", "ETH"),
        ("Kinshasa", "COD"),
    ]

    print("\nTest cities:")
    print("-" * 80)
    for city, country in test_cities:
        try:
            crs = mapper.get_crs_for_city(city, country)
            info = mapper.get_crs_info(city, country)
            has_override = mapper.has_city_override(city)

            print(f"\n{city}, {country}:")
            print(f"  CRS: {crs}")
            print(f"  Name: {info['name']}")
            print(f"  Type: {info['type']}")
            print(f"  Has override: {has_override}")
            if has_override:
                print(f"  Reason: {info['reason']}")
        except Exception as e:
            print(f"\n{city}, {country}: ERROR - {e}")

    print("\n" + "=" * 80)
    print("Metadata:")
    print("-" * 80)
    metadata = mapper.get_metadata()
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    print("\n✅ Test complete!")
