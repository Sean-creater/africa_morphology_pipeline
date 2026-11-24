"""Image-dataset construction pipeline for African urban morphology grid cells.

Stages: `sampling` (morphology-stratified cell sampling), `satellite_download`
(Sentinel-2 imagery via Google Earth Engine), `render` (the two PNG output
styles), and the `annotation_pairs` subpackage (anchor-based comparison-pair
dataset construction for human similarity annotation).
"""
