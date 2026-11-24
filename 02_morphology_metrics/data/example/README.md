# Example: Ndioum, Senegal

`Ndioum_Senegal/` contains a real (not synthetic) example linking the
two pipeline stages: `../../../01_hex_grid_workflow/` builds the base
grid, this directory's pipeline adds building, spatial, street, and
connectivity metrics to produce the final attribute table and Shapefile.

## Reproducing this example

```bash
python -m examples.reproduce_ndioum_full \
  --grid-project /path/to/africa_hex_grid_workflow \
  --output-dir /path/to/output
```

runs both projects end-to-end from the bundled inputs. To run only the
morphology stage against the bundled grid:

```bash
python -m examples.reproduce_ndioum --output-dir /path/to/output
```

## Third-party data attribution

The bundled inputs contain clipped or extracted third-party data. Retain
these attributions when redistributing this example:

- Building footprints: Google Open Buildings v3, CC BY 4.0.
- Roads: © OpenStreetMap contributors, ODbL 1.0.
- Building height and volume: DLR World Settlement Footprint 3D, CC BY 4.0.
- Population raster: WorldPop, CC BY 4.0.
- Land-cover rasters: contains modified Copernicus Service information.

See the manuscript's Supplementary Information for full citation details,
input provenance, and the reproduction verification results.
