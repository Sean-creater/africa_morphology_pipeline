"""Regression for WSF3D zero and sentinel pixels in batched zonal statistics."""

import numpy as np
import rasterio
import tempfile
import unittest
from pathlib import Path
from rasterio.transform import from_origin
from shapely.geometry import box

from africa_hex_grid_workflow.raster_extract import _process_grid_cell_stable


class RasterNoDataTest(unittest.TestCase):
    def test_volume_statistics_exclude_zero_and_sentinel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raster = Path(temp_dir) / 'volume.tif'
            profile = {
                'driver': 'GTiff', 'width': 2, 'height': 2, 'count': 1,
                'dtype': 'int32', 'crs': 'EPSG:32628',
                'transform': from_origin(0, 2, 1, 1), 'nodata': -32767,
            }
            with rasterio.open(raster, 'w', **profile) as dst:
                dst.write(np.array([[0, 10], [20, -32767]], dtype=np.int32), 1)

            stats = _process_grid_cell_stable((
                box(0, 0, 2, 2), str(raster), ['sum', 'mean', 'std'],
                [-32767.0, 0], True, True, (0.1, 1e9),
            ))
            self.assertEqual(stats, {'sum': 30.0, 'mean': 15.0, 'std': 5.0})


if __name__ == '__main__':
    unittest.main()
