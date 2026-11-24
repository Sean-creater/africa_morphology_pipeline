# Example output (SYNTHETIC — not real data)

Every file in this directory whose name contains `SYNTHETIC` is a fabricated
toy dataset. It exists only to show the schema (column names, dtypes, file
formats) that the pipeline produces for a real city. Building heights,
volumes, population counts, and land cover percentages in these files are
random numbers, not measurements — do not use them for any analysis.

Files:

- `toyville_SYNTHETIC_grid_250m_hex_enriched.{shp,dbf,shx,prj,cpg,csv,geojson}`
  — the per-hexagon grid with every attached data column.
- `toyville_SYNTHETIC_grid_boundary.geojson` — the dissolved outline of all
  hexagons, the artifact handed off to other modules.

See the top-level README for how to generate a real (non-synthetic) example.
