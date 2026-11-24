"""
Registries for two related data-coverage bookkeeping problems that arise
when processing thousands of cities unattended:

- ``NoBuildingDataTracker`` records individual cities where Google Open
  Buildings extraction returned zero buildings, so a resumed run does not
  re-attempt a known-empty city, and so coverage gaps are auditable after
  the fact.
- ``ExcludedCountriesTracker`` records whole countries that have been
  removed from the active processing plan after ``NoBuildingDataTracker``
  detects a run of consecutive no-data cities from the start of that
  country's city list, which is a strong signal that Open Buildings simply
  has no coverage there (rather than those particular cities being empty).

Both persist to JSON so state survives pipeline restarts/resumes.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class NoBuildingDataTracker:
    """Track and manage cities with no building data available."""

    def __init__(self, tracker_file: Path):
        """
        Initialize the tracker.

        Parameters
        ----------
        tracker_file : Path
            Path to the JSON file storing the tracking data
        """
        self.tracker_file = Path(tracker_file)
        self.data = self._load_data()
        # Track consecutive failures per country for exclusion logic
        self.country_consecutive_failures = {}
        # Track if country has had any successful city (resets exclusion eligibility)
        self.country_has_success = {}

    def _load_data(self) -> Dict:
        """Load existing tracking data or create new structure."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Load consecutive failures from file (for persistence across restarts)
                    if "consecutive_failures_by_country" in data:
                        self.country_consecutive_failures = data["consecutive_failures_by_country"]
                    # Load country success tracking
                    if "country_has_success" in data:
                        self.country_has_success = data["country_has_success"]
                    return data
            except Exception as e:
                logger.warning(f"Failed to load tracker file: {e}, creating new one")

        return {
            "metadata": {
                "description": "Cities with no Google Open Buildings data available",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_cities": 0,
                "affected_countries": []
            },
            "cities": {},
            "countries_summary": {},
            "consecutive_failures_by_country": {},  # Persistent storage for consecutive failures
            "country_has_success": {}  # Track if country has had any successful city
        }

    def _save_data(self):
        """Save tracking data to file."""
        try:
            self.data["metadata"]["last_updated"] = datetime.now().isoformat()
            # Save consecutive failures to file for persistence across restarts
            self.data["consecutive_failures_by_country"] = self.country_consecutive_failures
            # Save country success tracking
            self.data["country_has_success"] = self.country_has_success
            with open(self.tracker_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            logger.info(f"Updated no-building-data tracker: {self.tracker_file}")
        except Exception as e:
            logger.error(f"Failed to save tracker file: {e}")

    def is_known_no_data_city(self, city_name: str, country_name: str) -> bool:
        """
        Check if a city is already known to have no building data.

        Parameters
        ----------
        city_name : str
            Name of the city
        country_name : str
            Name of the country

        Returns
        -------
        bool
            True if city is known to have no data
        """
        city_key = f"{country_name}:{city_name}"
        return city_key in self.data["cities"]

    def add_city(
        self,
        city_name: str,
        country_name: str,
        iso3: str,
        retry_count: int = 0,
        boundary_area_km2: Optional[float] = None,
        coordinates: Optional[Dict] = None,
        notes: str = ""
    ):
        """
        Add a city to the no-data registry.

        Parameters
        ----------
        city_name : str
            Name of the city
        country_name : str
            Name of the country
        iso3 : str
            ISO3 country code
        retry_count : int
            Number of retry attempts made
        boundary_area_km2 : float, optional
            Area of city boundary in km2
        coordinates : dict, optional
            Boundary coordinates (lon_min, lat_min, lon_max, lat_max)
        notes : str
            Additional notes
        """
        city_key = f"{country_name}:{city_name}"

        # Check if already exists
        if city_key in self.data["cities"]:
            logger.info(f"City {city_key} already in no-data registry, updating...")
            self.data["cities"][city_key]["retry_count"] = retry_count
            self.data["cities"][city_key]["last_checked"] = datetime.now().isoformat()
            if notes:
                self.data["cities"][city_key]["notes"] = notes
        else:
            # Add new entry
            self.data["cities"][city_key] = {
                "city_name": city_name,
                "country_name": country_name,
                "iso3": iso3,
                "first_detected": datetime.now().isoformat(),
                "last_checked": datetime.now().isoformat(),
                "retry_count": retry_count,
                "boundary_area_km2": boundary_area_km2,
                "coordinates": coordinates,
                "notes": notes
            }

            # Update country summary
            if country_name not in self.data["countries_summary"]:
                self.data["countries_summary"][country_name] = {
                    "iso3": iso3,
                    "cities_count": 0,
                    "cities": []
                }

            self.data["countries_summary"][country_name]["cities_count"] += 1
            self.data["countries_summary"][country_name]["cities"].append(city_name)

            # Update metadata
            self.data["metadata"]["total_cities"] = len(self.data["cities"])
            self.data["metadata"]["affected_countries"] = list(self.data["countries_summary"].keys())

            logger.warning(f"Added {city_key} to no-building-data registry")

        self._save_data()

    def get_affected_countries(self) -> list:
        """Get list of countries with no-data cities."""
        return self.data["metadata"]["affected_countries"]

    def get_city_count(self) -> int:
        """Get total count of no-data cities."""
        return self.data["metadata"]["total_cities"]

    def get_country_cities(self, country_name: str) -> list:
        """Get list of no-data cities for a specific country."""
        if country_name in self.data["countries_summary"]:
            return self.data["countries_summary"][country_name]["cities"]
        return []

    def check_country_consecutive_failures(self, iso3: str, city_name: str) -> Dict:
        """
        Check if a country has consecutive city failures from the first city and should be excluded.

        Only consecutive failures counted from the FIRST city of the country
        trigger exclusion. If any city in the country has succeeded, the
        country is never excluded. Ten consecutive failures from the start
        of the country's city list is treated as strong evidence that Open
        Buildings has no coverage there, distinct from ordinary scattered
        coverage gaps.

        This design deliberately avoids two failure modes:
        1. Early cities succeed but later ones fail (uneven coverage within
           a country, e.g. observed for Egypt) - such a country should stay
           active.
        2. Resuming from a checkpoint skips already-completed cities, which
           would otherwise make unrelated failures look artificially
           consecutive.

        Parameters
        ----------
        iso3 : str
            ISO3 country code
        city_name : str
            Current city name that failed

        Returns
        -------
        dict
            Dictionary with keys:
            - should_exclude (bool): True if country should be excluded
            - consecutive_count (int): Number of consecutive failures
            - failed_cities (list): List of consecutively failed cities
        """
        # Check if this country has ever had a successful city
        if iso3 in self.country_has_success and self.country_has_success[iso3]:
            # Country has had at least one successful city, should NOT be excluded
            # Still track the failure for statistics, but don't count towards exclusion
            return {
                "should_exclude": False,
                "consecutive_count": 0,
                "failed_cities": [],
                "reason": "Country has had successful cities - not eligible for exclusion"
            }

        # Initialize tracking for this country if not exists
        if iso3 not in self.country_consecutive_failures:
            self.country_consecutive_failures[iso3] = []

        # Add current city to consecutive failures (only if not already in the list)
        if city_name not in self.country_consecutive_failures[iso3]:
            self.country_consecutive_failures[iso3].append(city_name)
            # Save immediately to persist the failure
            self._save_data()

        consecutive_count = len(self.country_consecutive_failures[iso3])
        should_exclude = consecutive_count >= 10  # Exclude after 10 consecutive failures from first city

        return {
            "should_exclude": should_exclude,
            "consecutive_count": consecutive_count,
            "failed_cities": self.country_consecutive_failures[iso3].copy()
        }

    def reset_country_consecutive_failures(self, iso3: str):
        """
        Reset consecutive failure counter for a country (called when a city succeeds).

        Also marks the country as having had a successful city, which
        permanently prevents it from being excluded in the future.

        Parameters
        ----------
        iso3 : str
            ISO3 country code
        """
        # Mark that this country has had a successful city
        self.country_has_success[iso3] = True

        # Clear consecutive failures
        if iso3 in self.country_consecutive_failures:
            failed_count = len(self.country_consecutive_failures[iso3])
            del self.country_consecutive_failures[iso3]
            logger.info(f"Country {iso3} had a successful city (cleared {failed_count} previous consecutive failures)")
            logger.info(f"  -> {iso3} is now permanently protected from exclusion")
        else:
            logger.info(f"Country {iso3} had a successful city (first success)")

        # Save immediately to persist the success status
        self._save_data()

    def get_summary(self) -> str:
        """Get a formatted summary of no-data cities."""
        total = self.get_city_count()
        countries = self.get_affected_countries()

        if total == 0:
            return "No cities with missing building data recorded."

        summary = [
            f"\n{'=' * 80}",
            f"CITIES WITH NO GOOGLE OPEN BUILDINGS DATA",
            f"{'=' * 80}",
            f"Total cities: {total}",
            f"Affected countries: {len(countries)}",
            f""
        ]

        for country in sorted(countries):
            cities = self.get_country_cities(country)
            iso3 = self.data["countries_summary"][country]["iso3"]
            summary.append(f"  {country} ({iso3}): {len(cities)} cities")
            for city in sorted(cities):
                summary.append(f"    - {city}")

        summary.append("=" * 80)
        return "\n".join(summary)


class ExcludedCountriesTracker:
    """Track and manage countries excluded from analysis due to no building data."""

    def __init__(self, tracker_file: Path):
        """
        Initialize the tracker.

        Parameters
        ----------
        tracker_file : Path
            Path to the JSON file storing the excluded countries data
        """
        self.tracker_file = Path(tracker_file)
        self.data = self._load_data()

    def _load_data(self) -> Dict:
        """Load existing tracking data or create new structure."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load excluded countries file: {e}, creating new one")

        return {
            "metadata": {
                "description": "Countries excluded from analysis due to lack of Open Buildings data",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_excluded_countries": 0,
                "total_excluded_cities": 0,
                "original_country_count": 55,  # ALL_AFRICA_ISO3 count (actual)
                "remaining_country_count": 55
            },
            "excluded_countries": {},
            "active_countries_iso3": []  # Updated list of countries still in analysis
        }

    def _save_data(self):
        """Save tracking data to file."""
        try:
            self.data["metadata"]["last_updated"] = datetime.now().isoformat()
            with open(self.tracker_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            logger.info(f"Updated excluded countries tracker: {self.tracker_file}")
        except Exception as e:
            logger.error(f"Failed to save excluded countries file: {e}")

    def is_country_excluded(self, iso3: str) -> bool:
        """
        Check if a country is excluded from analysis.

        Parameters
        ----------
        iso3 : str
            ISO3 country code

        Returns
        -------
        bool
            True if country is excluded
        """
        return iso3 in self.data["excluded_countries"]

    def exclude_country(
        self,
        iso3: str,
        country_name: str,
        failed_cities: List[str],
        total_cities_in_country: int,
        reason: str = "Consecutive cities with no Open Buildings data"
    ):
        """
        Exclude a country from analysis.

        Parameters
        ----------
        iso3 : str
            ISO3 country code
        country_name : str
            Country name
        failed_cities : list
            List of cities that had no building data
        total_cities_in_country : int
            Total number of cities in this country
        reason : str
            Reason for exclusion
        """
        if iso3 in self.data["excluded_countries"]:
            logger.warning(f"Country {country_name} ({iso3}) already excluded")
            return

        # Add to excluded countries
        self.data["excluded_countries"][iso3] = {
            "country_name": country_name,
            "iso3": iso3,
            "excluded_at": datetime.now().isoformat(),
            "reason": reason,
            "failed_cities": failed_cities,
            "total_cities_in_country": total_cities_in_country,
            "cities_checked_before_exclusion": len(failed_cities)
        }

        # Update metadata
        self.data["metadata"]["total_excluded_countries"] = len(self.data["excluded_countries"])
        self.data["metadata"]["total_excluded_cities"] += total_cities_in_country
        self.data["metadata"]["remaining_country_count"] = (
            self.data["metadata"]["original_country_count"] -
            self.data["metadata"]["total_excluded_countries"]
        )

        # Update active countries list (remove excluded country)
        if iso3 in self.data["active_countries_iso3"]:
            self.data["active_countries_iso3"].remove(iso3)

        logger.warning(f"Excluded country {country_name} ({iso3}) from analysis. "
                      f"Failed cities: {failed_cities}, Total cities in country: {total_cities_in_country}")

        self._save_data()

    def get_excluded_countries(self) -> Dict:
        """Get dictionary of all excluded countries."""
        return self.data["excluded_countries"]

    def get_excluded_country_count(self) -> int:
        """Get count of excluded countries."""
        return self.data["metadata"]["total_excluded_countries"]

    def get_remaining_country_count(self) -> int:
        """Get count of remaining active countries."""
        return self.data["metadata"]["remaining_country_count"]

    def get_total_excluded_cities(self) -> int:
        """Get total count of cities in excluded countries."""
        return self.data["metadata"]["total_excluded_cities"]

    def initialize_active_countries(self, all_africa_iso3: List[str]):
        """
        Initialize the active countries list with all African countries.

        Parameters
        ----------
        all_africa_iso3 : list
            List of all African country ISO3 codes from config
        """
        if not self.data["active_countries_iso3"]:
            self.data["active_countries_iso3"] = all_africa_iso3.copy()
            self.data["metadata"]["original_country_count"] = len(all_africa_iso3)
            self.data["metadata"]["remaining_country_count"] = len(all_africa_iso3)
            self._save_data()
            logger.info(f"Initialized active countries list with {len(all_africa_iso3)} countries")

    def get_active_countries_iso3(self) -> List[str]:
        """Get list of active (not excluded) country ISO3 codes."""
        return self.data["active_countries_iso3"]

    def get_summary(self) -> str:
        """Get a formatted summary of excluded countries."""
        total_excluded = self.get_excluded_country_count()
        remaining = self.get_remaining_country_count()
        excluded_cities = self.get_total_excluded_cities()

        if total_excluded == 0:
            return "No countries have been excluded from analysis."

        summary = [
            f"\n{'=' * 80}",
            f"COUNTRIES EXCLUDED FROM ANALYSIS (NO OPEN BUILDINGS DATA)",
            f"{'=' * 80}",
            f"Total excluded countries: {total_excluded}",
            f"Remaining active countries: {remaining}",
            f"Total cities in excluded countries: {excluded_cities}",
            f""
        ]

        for iso3, info in sorted(self.data["excluded_countries"].items()):
            summary.append(f"  {info['country_name']} ({iso3})")
            summary.append(f"    - Total cities: {info['total_cities_in_country']}")
            summary.append(f"    - Checked before exclusion: {info['cities_checked_before_exclusion']}")
            summary.append(f"    - Failed cities: {', '.join(info['failed_cities'])}")
            summary.append(f"    - Excluded at: {info['excluded_at']}")
            summary.append(f"    - Reason: {info['reason']}")
            summary.append("")

        summary.append("=" * 80)
        return "\n".join(summary)
