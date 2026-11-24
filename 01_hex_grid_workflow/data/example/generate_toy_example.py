#!/usr/bin/env python3
"""Generate the SYNTHETIC toy example grid checked into this directory.

WARNING: every value in the generated output (building height/volume,
population, land cover percentages) is fabricated with a fixed random seed,
purely to illustrate the output schema described in the top-level README. It
is NOT derived from any real raster data and must never be used for
analysis. It exists so a reader can inspect the CSV/GeoJSON/Shapefile column
layout, dtypes, and file structure without needing to obtain and process
115GB of real raw rasters first.

For a real example generated from actual data, run
`africa_hex_grid_workflow.pipeline` against your own copy of the input
sources (see the top-level README's "Reproducing an example city" section).
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from africa_hex_grid_workflow import config as cfg
from africa_hex_grid_workflow.diversity import calculate_shannon_diversity
from africa_hex_grid_workflow.hexgrid import HexagonalGridGenerator

OUTPUT_DIR = Path(__file__).parent
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)

    hex_gen = HexagonalGridGenerator(side_length=250)
    # A small bounding box, arbitrary in a UTM-like CRS, just large enough to
    # produce a few dozen hexagons — not tied to any real city's extent.
    hexagons = hex_gen.get_hexagons_in_bounds(600_000, 700_000, 602_000, 702_000, 0, 0)

    grid = gpd.GeoDataFrame(
        {
            "GridID": [f"TOY_0001_{i:06d}" for i in range(len(hexagons))],
            "geometry": [h["geometry"] for h in hexagons],
            "center_x": [h["center_x"] for h in hexagons],
            "center_y": [h["center_y"] for h in hexagons],
        },
        crs="EPSG:32630",
    )
    grid["City"] = "Toyville"
    grid["Area_m2"] = grid.geometry.area.round(2)
    grid["Center_X"] = grid["center_x"].round(2)
    grid["Center_Y"] = grid["center_y"].round(2)
    grid = grid.drop(columns=["center_x", "center_y"])

    n = len(grid)

    grid["H_max"] = np.round(rng.uniform(2, 25, n), 2).astype("float32")
    grid["H_mean"] = np.round(grid["H_max"] * rng.uniform(0.3, 0.7, n), 2).astype("float32")
    grid["H_std"] = np.round(grid["H_mean"] * rng.uniform(0.1, 0.4, n), 2).astype("float32")
    grid["H_sum"] = np.round(grid["H_mean"] * rng.uniform(50, 400, n), 2).astype("float32")

    grid["V_sum"] = np.round(grid["H_sum"] * rng.uniform(20, 60, n), 2).astype("float32")
    grid["V_mean"] = np.round(grid["V_sum"] / rng.uniform(50, 400, n), 2).astype("float32")
    grid["V_max"] = np.round(grid["V_mean"] * rng.uniform(2, 5, n), 2).astype("float32")
    grid["V_std"] = np.round(grid["V_mean"] * rng.uniform(0.2, 0.6, n), 2).astype("float32")

    grid["Pop2022"] = np.round(rng.uniform(0, 3000, n), 2).astype("float32")

    for col in cfg.LAND_COVER_COLUMN_NAMES.values():
        grid[col] = 0.0
    active_cols = ["LC_Built", "LC_Crop", "LC_Shrub", "LC_Herb", "LC_Bare", "LC_Water"]
    raw = rng.dirichlet(np.ones(len(active_cols)), size=n) * 100
    for i, col in enumerate(active_cols):
        grid[col] = np.round(raw[:, i], 2).astype("float32")

    grid = calculate_shannon_diversity(grid, precision=2)

    grid["LCF_Bare"] = np.round(grid["LC_Bare"] + rng.uniform(-5, 5, n), 2).clip(0, 100).astype("float32")
    grid["LCF_Built"] = np.round(grid["LC_Built"] + rng.uniform(-5, 5, n), 2).clip(0, 100).astype("float32")
    grid["LCF_Crop"] = np.round(grid["LC_Crop"] + rng.uniform(-5, 5, n), 2).clip(0, 100).astype("float32")
    grid["LCF_Grass"] = np.round(grid["LC_Herb"] + rng.uniform(-5, 5, n), 2).clip(0, 100).astype("float32")
    grid["LCF_Shrub"] = np.round(grid["LC_Shrub"] + rng.uniform(-5, 5, n), 2).clip(0, 100).astype("float32")
    grid["LCF_Tree"] = np.round(rng.uniform(0, 20, n), 2).astype("float32")
    grid["LCF_Water"] = np.round(grid["LC_Water"] + rng.uniform(-2, 2, n), 2).clip(0, 100).astype("float32")
    grid["LCF_SeaWtr"] = 0.0
    grid["LCF_Moss"] = 0.0
    grid["LCF_Snow"] = 0.0

    base = OUTPUT_DIR / "toyville_SYNTHETIC_grid_250m_hex_enriched"
    grid.to_file(base.with_suffix(".shp"))
    grid.drop(columns=["geometry"]).to_csv(base.with_suffix(".csv"), index=False)
    grid.to_file(base.with_suffix(".geojson"), driver="GeoJSON")

    grid_union = grid.geometry.union_all() if hasattr(grid.geometry, "union_all") else grid.geometry.unary_union
    boundary = gpd.GeoDataFrame(
        {
            "name": ["toyville_SYNTHETIC_grid_boundary"],
            "grid_cells": [len(grid)],
            "total_area_m2": [grid_union.area],
            "geometry": [grid_union],
        },
        crs=grid.crs,
    )
    boundary.to_file(OUTPUT_DIR / "toyville_SYNTHETIC_grid_boundary.geojson", driver="GeoJSON")

    print(f"Wrote {len(grid)} synthetic hexagons to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
