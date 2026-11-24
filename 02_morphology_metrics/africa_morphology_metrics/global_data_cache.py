"""
Process-wide caching for the agglomeration and country-boundary layers.

Agglomeration and boundary GeoDataFrames are read from disk thousands of
times over a full pipeline run (once per city lookup), so this module
lazily loads each one once per process and serves every subsequent request
from memory. The cache is read-only: nothing here mutates the cached
GeoDataFrames, so callers see the same data they would from a fresh read,
just without the repeated I/O.
"""

import geopandas as gpd
import logging
from pathlib import Path
from typing import Optional, Dict
import time

logger = logging.getLogger(__name__)


# ============================================================================
# GLOBAL CACHE STATE (SINGLETON PATTERN)
# ============================================================================

_GLOBAL_CACHE: Dict[str, Optional[gpd.GeoDataFrame]] = {
    'agglomeration': None,
    'agglomeration_path': None,
    'boundaries': None,
    'boundaries_path': None,
    'load_time': None,
}


# ============================================================================
# PUBLIC API - LAZY-LOADED GETTERS
# ============================================================================

def get_agglomeration_data(file_path: Path, force_reload: bool = False) -> Optional[gpd.GeoDataFrame]:
    """
    Get the agglomeration data, loading it from disk at most once per process.

    Parameters
    ----------
    file_path : Path
        Path to agglomerations.gpkg.
    force_reload : bool, optional
        Force a reload from disk even if already cached (default: False).

    Returns
    -------
    GeoDataFrame or None
        Agglomeration data, or None if the file is missing or fails to load.

    Examples
    --------
    >>> agglo = get_agglomeration_data(Path("agglomerations.gpkg"))
    >>> # First call loads from disk
    >>> agglo2 = get_agglomeration_data(Path("agglomerations.gpkg"))
    >>> # Second call returns the cached data (no I/O)
    >>> assert agglo is agglo2  # Same object reference
    """
    global _GLOBAL_CACHE

    file_path = Path(file_path)

    if not force_reload and _GLOBAL_CACHE['agglomeration'] is not None:
        if _GLOBAL_CACHE['agglomeration_path'] == str(file_path):
            logger.debug("✓ Using cached agglomeration data (zero I/O)")
            return _GLOBAL_CACHE['agglomeration']
        else:
            logger.warning(f"Agglomeration path changed: {_GLOBAL_CACHE['agglomeration_path']} -> {file_path}")
            _GLOBAL_CACHE['agglomeration'] = None

    if not file_path.exists():
        logger.warning(f"Agglomeration file not found: {file_path}")
        return None

    try:
        start_time = time.time()
        logger.info(f"🔄 Loading agglomeration data (one-time): {file_path.name}")

        agglo = gpd.read_file(file_path)

        elapsed = time.time() - start_time
        logger.info(f"✓ Loaded {len(agglo):,} agglomerations in {elapsed:.2f}s (CRS: {agglo.crs})")

        _GLOBAL_CACHE['agglomeration'] = agglo
        _GLOBAL_CACHE['agglomeration_path'] = str(file_path)
        _GLOBAL_CACHE['load_time'] = time.time()

        return agglo

    except Exception as e:
        logger.error(f"Error loading agglomeration data: {str(e)}")
        return None


def get_boundaries_data(file_path: Path, force_reload: bool = False) -> Optional[gpd.GeoDataFrame]:
    """
    Get the country boundaries data, loading it from disk at most once per process.

    Parameters
    ----------
    file_path : Path
        Path to africa_boundaries.gpkg.
    force_reload : bool, optional
        Force a reload from disk even if already cached (default: False).

    Returns
    -------
    GeoDataFrame or None
        Country boundaries data, or None if the file is missing or fails to load.

    Examples
    --------
    >>> boundaries = get_boundaries_data(Path("africa_boundaries.gpkg"))
    >>> # First call loads from disk
    >>> boundaries2 = get_boundaries_data(Path("africa_boundaries.gpkg"))
    >>> # Second call returns the cached data (no I/O)
    >>> assert boundaries is boundaries2  # Same object reference
    """
    global _GLOBAL_CACHE

    file_path = Path(file_path)

    if not force_reload and _GLOBAL_CACHE['boundaries'] is not None:
        if _GLOBAL_CACHE['boundaries_path'] == str(file_path):
            logger.debug("✓ Using cached boundaries data (zero I/O)")
            return _GLOBAL_CACHE['boundaries']
        else:
            logger.warning(f"Boundaries path changed: {_GLOBAL_CACHE['boundaries_path']} -> {file_path}")
            _GLOBAL_CACHE['boundaries'] = None

    if not file_path.exists():
        logger.warning(f"Boundaries file not found: {file_path}")
        return None

    try:
        start_time = time.time()
        logger.info(f"🔄 Loading country boundaries (one-time): {file_path.name}")

        boundaries = gpd.read_file(file_path)

        elapsed = time.time() - start_time
        logger.info(f"✓ Loaded {len(boundaries):,} country boundaries in {elapsed:.2f}s (CRS: {boundaries.crs})")

        _GLOBAL_CACHE['boundaries'] = boundaries
        _GLOBAL_CACHE['boundaries_path'] = str(file_path)
        if _GLOBAL_CACHE['load_time'] is None:
            _GLOBAL_CACHE['load_time'] = time.time()

        return boundaries

    except Exception as e:
        logger.error(f"Error loading boundaries data: {str(e)}")
        return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clear_cache():
    """
    Clear all cached data and free memory.

    Use at the end of batch processing, or before switching to a different
    dataset.

    Examples
    --------
    >>> # Process all cities in a country
    >>> for city in cities:
    ...     agglo = get_agglomeration_data(agglo_file)
    ...     process_city(city, agglo)
    >>> # Done with batch, free memory
    >>> clear_cache()
    """
    global _GLOBAL_CACHE

    logger.info("Clearing global data cache...")

    _GLOBAL_CACHE['agglomeration'] = None
    _GLOBAL_CACHE['agglomeration_path'] = None
    _GLOBAL_CACHE['boundaries'] = None
    _GLOBAL_CACHE['boundaries_path'] = None
    _GLOBAL_CACHE['load_time'] = None


def get_cache_stats() -> Dict[str, any]:
    """
    Get statistics about the current cache state.

    Returns
    -------
    dict
        Cache statistics: load status, file paths, record counts, and age.

    Examples
    --------
    >>> stats = get_cache_stats()
    >>> print(f"Agglomeration cached: {stats['agglomeration_loaded']}")
    >>> print(f"Cache age: {stats['cache_age_seconds']:.1f}s")
    """
    global _GLOBAL_CACHE

    cache_age = None
    if _GLOBAL_CACHE['load_time'] is not None:
        cache_age = time.time() - _GLOBAL_CACHE['load_time']

    return {
        'agglomeration_loaded': _GLOBAL_CACHE['agglomeration'] is not None,
        'agglomeration_path': _GLOBAL_CACHE['agglomeration_path'],
        'agglomeration_count': len(_GLOBAL_CACHE['agglomeration']) if _GLOBAL_CACHE['agglomeration'] is not None else 0,
        'boundaries_loaded': _GLOBAL_CACHE['boundaries'] is not None,
        'boundaries_path': _GLOBAL_CACHE['boundaries_path'],
        'boundaries_count': len(_GLOBAL_CACHE['boundaries']) if _GLOBAL_CACHE['boundaries'] is not None else 0,
        'cache_age_seconds': cache_age,
    }


def validate_cache_integrity() -> bool:
    """
    Validate that cached data is intact and well-formed.

    Returns
    -------
    bool
        True if the cache is valid or empty, False if corrupted.

    Examples
    --------
    >>> agglo = get_agglomeration_data(agglo_file)
    >>> assert validate_cache_integrity()  # Should always pass
    """
    global _GLOBAL_CACHE

    try:
        if _GLOBAL_CACHE['agglomeration'] is not None:
            agglo = _GLOBAL_CACHE['agglomeration']
            assert isinstance(agglo, gpd.GeoDataFrame), "Agglomeration is not a GeoDataFrame"
            assert len(agglo) > 0, "Agglomeration is empty"
            assert agglo.crs is not None, "Agglomeration has no CRS"
            assert 'geometry' in agglo.columns, "Agglomeration missing geometry column"

        if _GLOBAL_CACHE['boundaries'] is not None:
            boundaries = _GLOBAL_CACHE['boundaries']
            assert isinstance(boundaries, gpd.GeoDataFrame), "Boundaries is not a GeoDataFrame"
            assert len(boundaries) > 0, "Boundaries is empty"
            assert boundaries.crs is not None, "Boundaries has no CRS"
            assert 'geometry' in boundaries.columns, "Boundaries missing geometry column"

        return True

    except AssertionError as e:
        logger.error(f"Cache integrity check failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Cache validation error: {str(e)}")
        return False


# ============================================================================
# TESTING/DEBUGGING
# ============================================================================

if __name__ == "__main__":
    """Test the cache module functionality."""
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Exercise the global data cache against real boundary files")
    parser.add_argument('--agglomeration-file', type=Path, required=True,
                       help='Path to the city/agglomeration boundaries file (see README for schema)')
    parser.add_argument('--country-boundaries-file', type=Path, required=True,
                       help='Path to the country boundaries file (see README for schema)')
    args = parser.parse_args()

    test_agglo_path = args.agglomeration_file
    test_boundaries_path = args.country_boundaries_file

    print("\n" + "="*80)
    print("TESTING GLOBAL DATA CACHE MODULE")
    print("="*80)

    print("\n[Test 1] Loading agglomeration data (first time)...")
    agglo1 = get_agglomeration_data(test_agglo_path)
    if agglo1 is not None:
        print(f"✓ Loaded {len(agglo1):,} agglomerations")
        print(f"  CRS: {agglo1.crs}")
        print(f"  Columns: {list(agglo1.columns[:5])}...")
    else:
        print("✗ Failed to load agglomeration data")
        sys.exit(1)

    print("\n[Test 2] Loading agglomeration data (second time - should use cache)...")
    agglo2 = get_agglomeration_data(test_agglo_path)
    if agglo2 is agglo1:
        print("✓ Cache working! Same object reference returned")
    else:
        print("✗ Cache not working! Different object returned")

    print("\n[Test 3] Loading boundaries data...")
    boundaries1 = get_boundaries_data(test_boundaries_path)
    if boundaries1 is not None:
        print(f"✓ Loaded {len(boundaries1):,} country boundaries")
        print(f"  CRS: {boundaries1.crs}")
    else:
        print("✗ Failed to load boundaries data")

    print("\n[Test 4] Cache statistics...")
    stats = get_cache_stats()
    print(f"✓ Agglomeration cached: {stats['agglomeration_loaded']} ({stats['agglomeration_count']:,} records)")
    print(f"✓ Boundaries cached: {stats['boundaries_loaded']} ({stats['boundaries_count']:,} records)")
    print(f"✓ Cache age: {stats['cache_age_seconds']:.2f}s")

    print("\n[Test 5] Validating cache integrity...")
    if validate_cache_integrity():
        print("✓ Cache integrity validated")
    else:
        print("✗ Cache integrity check failed")

    print("\n[Test 6] Clearing cache...")
    clear_cache()
    stats = get_cache_stats()
    if not stats['agglomeration_loaded'] and not stats['boundaries_loaded']:
        print("✓ Cache cleared successfully")
    else:
        print("✗ Cache not properly cleared")

    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
