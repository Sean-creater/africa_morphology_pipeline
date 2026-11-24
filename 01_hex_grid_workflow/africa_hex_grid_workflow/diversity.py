"""Shannon land cover diversity index."""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np

logger = logging.getLogger(__name__)


def calculate_shannon_diversity(grid: gpd.GeoDataFrame, precision: int = 2) -> gpd.GeoDataFrame:
    """Add an `LC_Shannon` column computed from the grid's `LC_*` percentage columns.

    Shannon index: H = -sum(p_i * ln(p_i)) over land cover classes i with
    p_i > 0, where p_i is each class's share of the cell (0-1 scale).

    Land cover percentages for a cell do not always sum to exactly 100 (e.g.
    a hexagon straddling the raster edge, or partially covered by NoData
    pixels), so proportions are renormalized to sum to 1 before computing H
    — this measures the *composition* of whatever land cover was observed,
    independent of how much of the cell had valid land cover data at all.
    Cells with no land cover data (all percentages 0) get Shannon = 0 rather
    than an undefined value.

    Args:
        grid: GeoDataFrame with one or more `LC_*` percentage columns
            (as produced by raster_extract.attach_land_cover). The
            `LC_Shannon` column itself, if already present, is excluded from
            the input columns.
        precision: Decimal places to round the result to.

    Returns:
        A new GeoDataFrame with `LC_Shannon` added (or unchanged, with a
        warning, if no `LC_*` columns are found).
    """
    grid = grid.copy()

    lc_columns = [col for col in grid.columns if col.startswith("LC_") and col != "LC_Shannon"]
    if not lc_columns:
        logger.warning("No LC_ columns found; cannot calculate Shannon index")
        return grid

    logger.info("Found %d LC_ columns for Shannon calculation", len(lc_columns))

    lc_data = grid[lc_columns].values.astype(np.float64)  # (n_rows, n_classes)
    row_sums = lc_data.sum(axis=1)
    valid_rows = row_sums > 0
    logger.info("Rows with valid LC data: %s / %s", f"{valid_rows.sum():,}", f"{len(grid):,}")

    row_sums_safe = np.where(row_sums > 0, row_sums, 1.0)
    proportions = lc_data / row_sums_safe[:, np.newaxis]

    # p * ln(p) -> 0 as p -> 0, so substituting 1.0 for zero proportions
    # before taking the log is safe: those terms are then multiplied by the
    # (unmodified) zero proportion and vanish.
    proportions_safe = np.where(proportions > 0, proportions, 1.0)
    p_ln_p = proportions * np.log(proportions_safe)

    shannon_values = -np.sum(p_ln_p, axis=1)
    shannon_values = np.where(row_sums > 0, shannon_values, 0.0)

    grid["LC_Shannon"] = np.round(shannon_values, precision)

    valid_shannon = grid["LC_Shannon"][grid["LC_Shannon"] > 0]
    if len(valid_shannon) > 0:
        logger.info(
            "LC_Shannon: min=%.3f, max=%.3f, mean=%.3f, median=%.3f",
            valid_shannon.min(), valid_shannon.max(), valid_shannon.mean(), valid_shannon.median(),
        )
        logger.info("Cells with diversity data: %s / %s", f"{len(valid_shannon):,}", f"{len(grid):,}")
    else:
        logger.warning("No valid Shannon values calculated")

    return grid
