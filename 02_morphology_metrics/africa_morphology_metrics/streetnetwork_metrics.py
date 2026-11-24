"""
Street Network Metrics Calculation Module
Contains functions for calculating street network and building-street metrics using momepy.

This module provides functions to:
1. Calculate street network intensity metrics per grid cell
2. Compute network topology metrics (nodes, intersections, deadends)
3. Calculate plot density metrics
4. Calculate street profile metrics (Dimension)
5. Calculate street orientation metrics (Spatial Distribution)
6. Calculate building-street alignment (Spatial Distribution)

Metrics calculated:
- Intensity: street_density, node_density, streets_per_node, deadend_density,
  intersection_density, segment_density, plot_density
- Dimension: street_width, street_width_deviation, street_openness
- Shape: street_linearity
- Spatial Distribution: street_orientation, building_street_alignment
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import momepy
import networkx as nx
import logging
import warnings
from pathlib import Path
from tqdm import tqdm
from shapely.ops import unary_union, polygonize
from shapely.prepared import prep
from typing import Optional
from multiprocessing import Pool, cpu_count
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logger
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='momepy')
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _remove_false_nodes_custom(G: nx.Graph) -> nx.Graph:
    """
    Remove degree-2 nodes (false nodes / vertices along a bend) from a network graph.

    A degree-2 node is a continuation point where a road bends but doesn't split.
    Removing these preserves road continuity for linearity calculations, and this
    custom implementation merges the connected LineString geometries so the merged
    edge's geometry (and therefore length and linearity) remains geometrically correct.

    Parameters:
    -----------
    G : nx.Graph or nx.MultiGraph
        Network graph with nodes as coordinate tuples and 'geometry' edge attribute

    Returns:
    --------
    nx.Graph
        Cleaned graph with degree-2 nodes removed and edges geometrically merged

    Notes:
    ------
    - Preserves all edge attributes
    - Merges LineString geometries when connecting edges
    - Only removes nodes with exactly degree 2
    """
    from shapely.geometry import LineString
    from shapely.ops import linemerge

    G_clean = G.copy()

    # Find all degree-2 nodes
    degree_2_nodes = [n for n, d in G_clean.degree() if d == 2]

    if len(degree_2_nodes) == 0:
        logger.info(f"  No false nodes (degree-2) found")
        return G_clean

    logger.info(f"  Found {len(degree_2_nodes):,} degree-2 nodes to remove")

    # Remove each degree-2 node by connecting its neighbors
    for node in degree_2_nodes:
        if node not in G_clean:
            # Already removed (can happen if nodes are connected)
            continue

        neighbors = list(G_clean.neighbors(node))

        if len(neighbors) != 2:
            # Skip if degree changed (shouldn't happen, but safety check)
            continue

        n1, n2 = neighbors

        # Get edge data and geometries from both edges
        edge_data = {}
        geom1 = None
        geom2 = None

        if isinstance(G_clean, nx.MultiGraph):
            # MultiGraph: handle multiple edges
            edges_to_n1 = list(G_clean[node][n1].values())
            edges_to_n2 = list(G_clean[node][n2].values())

            if edges_to_n1:
                edge_data = edges_to_n1[0].copy() if edges_to_n1[0] else {}
                geom1 = edge_data.get('geometry')
            if edges_to_n2:
                if not edge_data:
                    edge_data = edges_to_n2[0].copy() if edges_to_n2[0] else {}
                geom2 = edges_to_n2[0].get('geometry') if edges_to_n2[0] else None
        else:
            # Simple Graph
            if G_clean.has_edge(node, n1):
                edge_data.update(G_clean[node][n1])
                geom1 = edge_data.get('geometry')
            if G_clean.has_edge(node, n2):
                edge_data.update(G_clean[node][n2])
                geom2 = G_clean[node][n2].get('geometry')

        # Merge geometries if they exist
        if geom1 and geom2:
            # Use shapely's linemerge to properly merge the two LineStrings
            # This handles orientation automatically
            try:
                merged = linemerge([geom1, geom2])
                if merged.geom_type == 'LineString':
                    edge_data['geometry'] = merged
                else:
                    # linemerge returns MultiLineString if they don't connect
                    # Fall back to manual coordinate merging
                    coords1 = list(geom1.coords)
                    coords2 = list(geom2.coords)
                    # Try to connect them (remove duplicate middle point)
                    if coords1[-1] == coords2[0]:
                        merged_coords = coords1 + coords2[1:]
                    elif coords1[0] == coords2[-1]:
                        merged_coords = coords2 + coords1[1:]
                    elif coords1[-1] == coords2[-1]:
                        merged_coords = coords1 + list(reversed(coords2))[1:]
                    elif coords1[0] == coords2[0]:
                        merged_coords = list(reversed(coords1)) + coords2[1:]
                    else:
                        # Just connect them
                        merged_coords = coords1 + coords2
                    edge_data['geometry'] = LineString(merged_coords)
            except Exception as e:
                # If merge fails, keep first geometry
                logger.warning(f"  Failed to merge geometries for node {node}: {e}")
                edge_data['geometry'] = geom1
        elif geom1:
            # Only first edge has geometry
            edge_data['geometry'] = geom1
        elif geom2:
            # Only second edge has geometry
            edge_data['geometry'] = geom2
        # else: no geometry to merge

        # Add merged edge
        G_clean.add_edge(n1, n2, **edge_data)

        # Remove the degree-2 node
        G_clean.remove_node(node)

    removed_count = len(degree_2_nodes) - sum(1 for n in degree_2_nodes if n in G_clean)
    logger.info(f"  Successfully removed {removed_count:,} false nodes and merged their geometries")

    return G_clean


def _project_crs(gdf: gpd.GeoDataFrame, target_crs: str = "ESRI:102022") -> gpd.GeoDataFrame:
    """
    Project GeoDataFrame to target CRS.

    NOTE: The default ESRI:102022 (Africa Albers) is a fallback. The main pipeline
    uses city-specific UTM zones via the projection_helper module. Each city gets
    its optimal UTM zone for accurate distance/area calculations.

    Parameters:
    -----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    target_crs : str
        Target coordinate reference system (default: "ESRI:102022" fallback)
        Pipeline uses city-specific UTM zones (e.g., "EPSG:32631" for Lagos)

    Returns:
    --------
    GeoDataFrame
        Reprojected GeoDataFrame
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame does not have CRS; please set CRS first (e.g., 'EPSG:4326')")

    if gdf.crs.to_string() != target_crs:
        return gdf.to_crs(target_crs)
    return gdf


def _clip_roads_to_study_area(roads: gpd.GeoDataFrame, study_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clip roads to study area and clean geometries.

    IMPORTANT: This function does NOT explode MultiLineString to preserve road geometry
    for linearity calculation. Exploding happens later in _split_lines_at_intersections.

    Parameters:
    -----------
    roads : GeoDataFrame
        Road network geometries
    study_area : GeoDataFrame
        Study area boundary (typically grid cells)

    Returns:
    --------
    GeoDataFrame
        Clipped and cleaned road network (may contain MultiLineString)
    """
    logger.info("  Clipping roads to study area...")

    # Create union polygon of study area and prepare it for faster operations
    study_poly = unary_union(study_area.geometry.values)
    prepared_study_poly = prep(study_poly)

    # Use prepared geometry for much faster intersection testing
    # This can be 10-100x faster for large datasets
    roads_clipped = roads[roads.geometry.apply(lambda g: prepared_study_poly.intersects(g))].copy()

    # DO NOT EXPLODE HERE - preserve MultiLineString for linearity calculation
    # Exploding will happen in _build_network_graph after linearity is calculated

    # Remove empty or zero-length geometries using vectorized operations
    is_valid = ~roads_clipped.geometry.is_empty & (roads_clipped.geometry.length > 0)
    roads_clipped = roads_clipped[is_valid].copy()

    logger.info(f"  Clipped to {len(roads_clipped):,} road features (may contain MultiLineString)")

    return roads_clipped


# ============================================================================
# NETWORK CONSTRUCTION AND METRICS
# ============================================================================

def _find_intersections_batch(args):
    """
    Worker function for parallel intersection finding.

    Parameters:
    -----------
    args : tuple
        (start_idx, end_idx, geom_list, tree)

    Returns:
    --------
    set
        Set of intersection point coordinates
    """
    from shapely.geometry import Point, MultiPoint

    start_idx, end_idx, geom_list, tree = args
    intersection_points_local = set()

    for idx in range(start_idx, end_idx):
        geom = geom_list[idx]
        # Use STRtree query for faster spatial lookup
        possible_matches_idx = tree.query(geom, predicate='intersects')

        for other_idx in possible_matches_idx:
            # Skip self and already processed pairs (symmetric)
            if other_idx <= idx:
                continue

            other_geom = geom_list[other_idx]

            # Get intersection
            intersection = geom.intersection(other_geom)

            if intersection.is_empty:
                continue

            # Extract points from intersection
            if isinstance(intersection, Point):
                # Round coordinates to avoid floating point issues
                coord = (round(intersection.x, 1), round(intersection.y, 1))
                intersection_points_local.add(coord)
            elif isinstance(intersection, MultiPoint):
                for pt in intersection.geoms:
                    coord = (round(pt.x, 1), round(pt.y, 1))
                    intersection_points_local.add(coord)

    return intersection_points_local


def _split_lines_at_intersections(roads: gpd.GeoDataFrame, n_jobs: int = None) -> gpd.GeoDataFrame:
    """
    Split LineStrings at ALL intersection points, including endpoints.

    This is essential for:
    1. Correct intersection counting (degree 3+ nodes)
    2. Accurate street network topology
    3. The resulting degree-2 nodes will be removed later by _remove_false_nodes_custom()

    Using unary_union (rather than only splitting at pre-existing shared vertices)
    ensures T-junctions and mid-segment crossings are captured, not just intersections
    that already coincide with a line endpoint.

    Parameters:
    -----------
    roads : GeoDataFrame
        Road network with LineString geometries
    n_jobs : int, optional
        Not used in this simplified version (kept for API compatibility)

    Returns:
    --------
    GeoDataFrame
        Road network with lines split at ALL intersection points
    """
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import unary_union

    logger.info("  Splitting roads at ALL intersections (using unary_union)...")
    logger.info(f"    Input: {len(roads):,} road features")

    # Step 1: Explode MultiLineString to LineString
    roads_exploded = roads.explode(index_parts=False, ignore_index=True)
    roads_exploded = roads_exploded[~roads_exploded.geometry.is_empty].copy()
    logger.info(f"    After exploding: {len(roads_exploded):,} road segments")

    # Step 2: Use unary_union to automatically split at ALL intersection points
    # This includes: T-junctions, crosses, endpoint overlaps, etc.
    logger.info("    Using unary_union to split at all intersections...")
    merged = unary_union(roads_exploded.geometry.tolist())

    # Step 3: Extract individual line segments
    if isinstance(merged, MultiLineString):
        split_geoms = list(merged.geoms)
    elif isinstance(merged, LineString):
        split_geoms = [merged]
    else:
        # Fallback: if unexpected geometry type, return original
        logger.warning(f"    Unexpected geometry type: {type(merged)}, returning original")
        return roads_exploded.copy()

    # Filter out very short segments (< 0.1m) that may be artifacts
    split_geoms = [g for g in split_geoms if g.length >= 0.1]

    # Create output GeoDataFrame
    roads_split = gpd.GeoDataFrame(
        {'geometry': split_geoms},
        crs=roads.crs
    )

    logger.info(f"    Output: {len(roads_split):,} road segments (split at all intersections)")
    logger.info(f"    Note: this includes degree-2 nodes which will be removed by _remove_false_nodes_custom()")

    return roads_split


def _build_network_graph(roads: gpd.GeoDataFrame) -> tuple:
    """
    Build primal network graph from road geometries using momepy.

    This function properly prepares the road network by:
    1. Ensuring clean geometries
    2. Building the network graph with proper intersection handling
    3. Removing false nodes (degree-2 nodes)

    The key to correct intersection counting is ensuring that the input road
    network has been properly split at all intersection points. If roads cross
    but don't have nodes at crossings, they won't be detected as intersections.

    Parameters:
    -----------
    roads : GeoDataFrame
        Road network geometries

    Returns:
    --------
    tuple
        (nodes_gdf, edges_gdf, graph) - Node GeoDataFrame, Edge GeoDataFrame, and NetworkX graph
    """
    logger.info("  Building network graph...")
    logger.info(f"  Initial road segments: {len(roads):,}")

    # Step 1: Split roads at true intersection points, so all road crossings
    # have nodes in the graph. Without this, two roads crossing in the middle
    # would not produce a junction node.
    roads_split = _split_lines_at_intersections(roads)
    logger.info(f"  After splitting at intersections: {len(roads_split):,} segments")

    # Step 2: Build primal graph (roads as edges, junctions as nodes)
    # Now all intersections are already endpoints, so momepy can connect them correctly
    G = momepy.gdf_to_nx(roads_split, approach="primal", directed=False)

    # Step 3: Clean network topology - remove degree-2 nodes.
    # This preserves road continuity for linearity calculation, keeps only true
    # network nodes (dead ends, intersections), and avoids breaking roads into
    # many small straight segments.
    logger.info("  Cleaning network topology (removing degree-2 nodes)...")
    initial_nodes = G.number_of_nodes()

    # Custom implementation is used instead of momepy.remove_false_nodes()
    # for compatibility with this pipeline's momepy version.
    G_clean = _remove_false_nodes_custom(G)

    final_nodes = G_clean.number_of_nodes()
    removed = initial_nodes - final_nodes
    if removed > 0:
        logger.info(f"  Nodes: {initial_nodes:,} -> {final_nodes:,} (removed {removed:,} false nodes)")
    else:
        logger.info(f"  No false nodes found, keeping all {final_nodes:,} nodes")

    # Step 4: Convert back to GeoDataFrames
    nodes_gdf, edges_gdf = momepy.nx_to_gdf(G_clean)

    # CRS is set explicitly because momepy.nx_to_gdf may not preserve it
    if roads.crs is not None:
        if nodes_gdf.crs is None:
            nodes_gdf = nodes_gdf.set_crs(roads.crs)
        if edges_gdf.crs is None:
            edges_gdf = edges_gdf.set_crs(roads.crs)

    # Step 5: Add node degree.
    # Graph nodes are identified by coordinate tuples (x, y), but
    # momepy.nx_to_gdf creates an integer nodeID column, so we map from
    # coordinate tuples to degrees explicitly.
    coord_to_degree = {}
    for node, degree in G_clean.degree():
        # node is a coordinate tuple like (0.0, 0.0)
        coord_to_degree[node] = degree

    # Vectorized mapping: create coordinate tuples and map to degrees
    # This is much faster than apply() for large datasets
    coord_tuples = list(zip(nodes_gdf['x'], nodes_gdf['y']))
    nodes_gdf["degree"] = [coord_to_degree.get(coord, 0) for coord in coord_tuples]

    # Rename nodeID to node_id for consistency
    if 'nodeID' in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.rename(columns={"nodeID": "node_id"})
    elif 'node_id' not in nodes_gdf.columns:
        # If no node_id column exists, create one
        nodes_gdf['node_id'] = range(len(nodes_gdf))

    # Step 6: Add edge length
    edges_gdf = edges_gdf.reset_index().rename(columns={"index": "edge_id"})
    edges_gdf["length_m"] = edges_gdf.geometry.length

    # Log degree distribution for debugging
    degree_counts = nodes_gdf['degree'].value_counts().sort_index()
    logger.info(f"  Graph created: {len(nodes_gdf):,} nodes, {len(edges_gdf):,} edges")
    logger.info(f"  Node degree distribution:")
    for deg, count in degree_counts.items():
        node_type = ""
        if deg == 1:
            node_type = " (dead ends)"
        elif deg == 2:
            node_type = " (continuation)"
        elif deg == 3:
            node_type = " (3-way intersections)"
        elif deg == 4:
            node_type = " (4-way intersections)"
        elif deg > 4:
            node_type = f" ({deg}-way intersections)"
        logger.info(f"    Degree {deg}: {count:,} nodes{node_type}")

    return nodes_gdf, edges_gdf, G_clean


def _calculate_street_lengths_per_cell(edges_gdf: gpd.GeoDataFrame,
                                        cells: gpd.GeoDataFrame,
                                        cell_id_col: str) -> tuple:
    """
    Calculate total street length and segment count per grid cell using overlay.

    Parameters:
    -----------
    edges_gdf : GeoDataFrame
        Street edges with geometries
    cells : GeoDataFrame
        Grid cells
    cell_id_col : str
        Column name for grid cell IDs

    Returns:
    --------
    tuple
        (length_by_cell, segments_by_cell) - Series of total length and segment count per cell
    """
    logger.info("  Calculating street lengths per cell...")

    # Ensure spatial index is built for faster operations
    if not hasattr(edges_gdf, '_sindex') or edges_gdf._sindex is None:
        _ = edges_gdf.sindex
    if not hasattr(cells, '_sindex') or cells._sindex is None:
        _ = cells.sindex

    # Overlay edges with cells
    edges_for_overlay = edges_gdf[["edge_id", "geometry"]]
    edges_in_cell = gpd.overlay(
        edges_for_overlay,
        cells[[cell_id_col, "geometry"]],
        how="intersection",
        keep_geom_type=True
    )

    # Calculate partial lengths using vectorized operation
    edges_in_cell["part_len_m"] = edges_in_cell.geometry.length

    # Aggregate by cell using groupby (already optimized in pandas)
    length_by_cell = edges_in_cell.groupby(cell_id_col)["part_len_m"].sum()
    segments_by_cell = edges_in_cell.groupby(cell_id_col)["edge_id"].nunique()

    return length_by_cell, segments_by_cell


def _calculate_node_metrics_per_cell(nodes_gdf: gpd.GeoDataFrame,
                                      cells: gpd.GeoDataFrame,
                                      cell_id_col: str) -> dict:
    """
    Calculate node-based metrics per grid cell using spatial join.

    Parameters:
    -----------
    nodes_gdf : GeoDataFrame
        Network nodes with degree information
    cells : GeoDataFrame
        Grid cells
    cell_id_col : str
        Column name for grid cell IDs

    Returns:
    --------
    dict
        Dictionary with Series for each metric:
        - node_count: Total number of meaningful nodes (degree != 2)
        - deadends: Number of dead ends (degree-1 nodes)
        - junctions: Number of intersections (degree >= 3 nodes)
        - three_way: Number of 3-way intersections (degree-3 nodes)
        - four_way: Number of 4-way intersections (degree-4 nodes)
    """
    logger.info("  Calculating node metrics per cell...")
    logger.info(f"    Nodes: {len(nodes_gdf)}, CRS: {nodes_gdf.crs}")
    logger.info(f"    Cells: {len(cells)}, CRS: {cells.crs}")

    if 'node_id' not in nodes_gdf.columns:
        logger.error("    node_id column missing from nodes_gdf!")
        logger.info(f"    Available columns: {list(nodes_gdf.columns)}")
        raise KeyError("nodes_gdf missing 'node_id' column")

    if 'degree' not in nodes_gdf.columns:
        logger.error("    degree column missing from nodes_gdf!")
        raise KeyError("nodes_gdf missing 'degree' column")

    # Check degree distribution
    degree_dist = nodes_gdf['degree'].value_counts().sort_index()
    logger.info(f"    Node degree distribution: {dict(degree_dist.head(10))}")

    # Ensure spatial indices are built for faster sjoin
    if not hasattr(nodes_gdf, '_sindex') or nodes_gdf._sindex is None:
        _ = nodes_gdf.sindex
    if not hasattr(cells, '_sindex') or cells._sindex is None:
        _ = cells.sindex

    # Spatial join nodes to cells
    # 'within' is used (rather than 'intersects') so that nodes exactly on a
    # cell boundary are not double-counted across neighboring cells.
    logger.info(f"    Performing spatial join...")
    nodes_in_cell = gpd.sjoin(
        nodes_gdf[["node_id", "geometry", "degree"]],
        cells[[cell_id_col, "geometry"]],
        predicate="within",
        how="inner"
    )

    logger.info(f"    Spatial join result: {len(nodes_in_cell)} matches (before dedup)")

    # Remove duplicate nodes (nodes on cell boundaries may still match multiple
    # cells); keep only the first cell match for each node.
    nodes_in_cell = nodes_in_cell.drop_duplicates(subset="node_id", keep="first")
    logger.info(f"    After deduplication: {len(nodes_in_cell)} matches")

    # Count nodes by type using vectorized operations.
    # Degree-2 nodes are excluded from node_count: they are internal vertices
    # along a single road (continuation points), not true network nodes.
    # Meaningful network nodes are:
    #   - Degree 1: dead ends (road terminals)
    #   - Degree >= 3: intersections (where roads meet)

    # Pre-filter by degree for better performance
    non_degree2 = nodes_in_cell["degree"] != 2
    degree1 = nodes_in_cell["degree"] == 1
    degree3plus = nodes_in_cell["degree"] >= 3
    degree3 = nodes_in_cell["degree"] == 3
    degree4 = nodes_in_cell["degree"] == 4

    node_count = nodes_in_cell.loc[non_degree2].groupby(cell_id_col)["node_id"].nunique()
    deadends = nodes_in_cell.loc[degree1].groupby(cell_id_col)["node_id"].nunique()
    junctions = nodes_in_cell.loc[degree3plus].groupby(cell_id_col)["node_id"].nunique()
    three_way = nodes_in_cell.loc[degree3].groupby(cell_id_col)["node_id"].nunique()
    four_way = nodes_in_cell.loc[degree4].groupby(cell_id_col)["node_id"].nunique()

    logger.info(f"    Statistics:")
    logger.info(f"      Cells with deadends: {len(deadends)} (total: {deadends.sum()})")
    logger.info(f"      Cells with junctions: {len(junctions)} (total: {junctions.sum()})")
    logger.info(f"      Cells with 3-way: {len(three_way)} (total: {three_way.sum()})")
    logger.info(f"      Cells with 4-way: {len(four_way)} (total: {four_way.sum()})")

    return {
        "node_count": node_count,
        "deadends": deadends,
        "junctions": junctions,
        "three_way": three_way,
        "four_way": four_way
    }


def _polygonize_single_cell(args):
    """
    Worker function for parallel polygonization of a single cell.

    Parameters:
    -----------
    args : tuple
        (cell_id, cell_geom, segs, min_area)

    Returns:
    --------
    tuple
        (cell_id, plot_count)
    """
    from shapely.geometry import LineString, MultiLineString

    cid, geom, segs, min_area = args

    # Create prepared geometry here (inside worker process)
    # to avoid pickling issues
    prepared_geom = prep(geom)

    # Add cell boundary as lines to close the network, so plots are formed
    # even for buildings/streets adjacent to a cell edge.
    all_lines = list(segs)  # Make a copy

    # Extract boundary of the cell and convert to LineString(s)
    boundary = geom.boundary
    if isinstance(boundary, LineString):
        all_lines.append(boundary)
    elif isinstance(boundary, MultiLineString):
        all_lines.extend(list(boundary.geoms))

    if not all_lines:
        return (cid, 0)

    # Merge all lines (roads + cell boundary) and polygonize
    try:
        merged = unary_union(all_lines)
        polys = list(polygonize(merged))

        # Filter to plots within cell using prepared geometry
        # Also filter out very small polygons (likely artifacts)
        polys_in = [p for p in polys
                    if p.area > min_area and prepared_geom.contains(p.centroid)]
        return (cid, len(polys_in))
    except Exception:
        # Silently fail for individual cells
        return (cid, 0)


def _calculate_plots_per_cell(edges_gdf: gpd.GeoDataFrame,
                               cells: gpd.GeoDataFrame,
                               cell_id_col: str,
                               n_jobs: int = None) -> pd.Series:
    """
    Calculate number of plots (polygonized areas) per grid cell using parallel processing.

    For each cell, this function:
    1. Clips roads to the cell boundary
    2. Adds the cell boundary itself as additional lines
    3. Polygonizes the combined network (roads + boundary)
    4. Counts resulting enclosed polygons within the cell

    This ensures that plots are properly formed even at cell edges.

    Parameters:
    -----------
    edges_gdf : GeoDataFrame
        Street edges with geometries
    cells : GeoDataFrame
        Grid cells
    cell_id_col : str
        Column name for grid cell IDs
    n_jobs : int, optional
        Number of parallel workers (default: cpu_count())

    Returns:
    --------
    Series
        Number of plots per cell
    """
    from shapely.geometry import LineString, MultiLineString

    logger.info("  Calculating plots per cell...")

    # Set default workers
    if n_jobs is None:
        n_jobs = cpu_count()

    # Ensure spatial indices are built
    if not hasattr(edges_gdf, '_sindex') or edges_gdf._sindex is None:
        _ = edges_gdf.sindex
    if not hasattr(cells, '_sindex') or cells._sindex is None:
        _ = cells.sindex

    # Overlay to get edges per cell
    edges_for_overlay = edges_gdf[["edge_id", "geometry"]]
    edges_in_cell = gpd.overlay(
        edges_for_overlay,
        cells[[cell_id_col, "geometry"]],
        how="intersection",
        keep_geom_type=True
    )

    # Group edges by cell - use dict for faster lookup
    geoms_by_cell = edges_in_cell.groupby(cell_id_col)["geometry"].apply(list).to_dict()

    # Prepare arguments for parallel processing
    min_area = 1.0  # Minimum plot area in square meters
    args_list = []
    cell_ids = []

    for cid, geom in cells[[cell_id_col, "geometry"]].itertuples(index=False):
        cell_ids.append(cid)
        segs = geoms_by_cell.get(cid, [])
        # Don't create prepared_geom here - it will be created in worker process
        args_list.append((cid, geom, segs, min_area))

    # Polygonize for each cell using parallel processing
    if n_jobs == 1:
        # Single-threaded mode
        results = []
        for args in tqdm(args_list, desc="    Polygonizing plots", unit="cell"):
            results.append(_polygonize_single_cell(args))
    else:
        # Multi-threaded mode
        logger.info(f"    Using {n_jobs} workers for parallel polygonization")
        with Pool(processes=n_jobs) as pool:
            results = list(tqdm(
                pool.imap(_polygonize_single_cell, args_list),
                total=len(args_list),
                desc="    Polygonizing plots",
                unit="cell"
            ))

    # Extract results
    result_dict = dict(results)
    plot_counts = [result_dict[cid] for cid in cell_ids]

    return pd.Series(plot_counts, index=cell_ids, name="plot_count")


# ============================================================================
# MAIN CALCULATION FUNCTION
# ============================================================================

def calculate_street_network_metrics(
    roads: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    buildings: Optional[gpd.GeoDataFrame] = None,
    cell_id_col: str = "grid_id",
    area_col: str = "Area_m2",
    target_crs: str = "ESRI:102022",
    n_jobs: int = None
) -> gpd.GeoDataFrame:
    """
    Calculate street network intensity metrics for grid cells.

    This is the main function that orchestrates the calculation of all street
    network metrics. It performs the following steps:
    1. Project data to target CRS
    2. Clip roads to study area
    3. Build network graph
    4. Create merged roads (semantic streets)
    5. Calculate street lengths and segments per cell
    6. Calculate node metrics per cell
    7. Calculate plots per cell (with parallel processing)
    8. Clip merged roads to cell boundaries
    9. Compute density and advanced metrics using the clipped roads

    Merged roads are clipped to cell boundaries before street-level metrics
    (StreetDens, StrWidth, StrOpen, StrLin, StrOrient) are computed, so that
    out-of-cell street portions never inflate a cell's values.

    All calculated metrics are of type "Intensity" following the momepy framework.

    Parameters:
    -----------
    roads : GeoDataFrame
        Road network geometries (LineString)
    cells : GeoDataFrame
        Grid cells (Polygon) with unique identifiers
    buildings : GeoDataFrame, optional
        Building footprints (Polygon), used for street profile and
        building-street alignment metrics; skipped if not provided
    cell_id_col : str
        Column name for grid cell IDs (default: "grid_id")
    area_col : str
        Column name for cell area in square meters (default: "Area_m2")
        If not present, will be calculated from geometry
    target_crs : str
        Target coordinate reference system (default: "ESRI:102022" fallback)
        NOTE: Main pipeline uses city-specific UTM zones via projection_helper
        for optimal accuracy (e.g., "EPSG:32631" for Lagos)
    n_jobs : int, optional
        Number of parallel workers for polygon computation (default: cpu_count())
        Set to 1 to disable parallel processing

    Returns:
    --------
    GeoDataFrame
        Input cells GeoDataFrame with added metric columns:
        - StreetDens: Street length per unit area (m/m²)
        - NodeDens: Nodes per unit area (count/m²)
        - StreetsPN: Average streets per node
        - SegmentDen: Segments per unit area (count/m²)
        - DeadendDen: Dead ends per unit area (count/m²)
        - InterDens: Intersections per unit area (count/m²)
        - Ratio3way: Ratio of 3-way to total intersections
        - Ratio4way: Ratio of 4-way to total intersections
        - CuldesacR: Ratio of dead ends to intersections
        - PlotDens: Plots per unit area (count/m²)
        - BldgDensN: Building density from network analysis (ratio)
        - PlotFrag: Plot fragmentation indicator (ratio)
        - StrWidth: Mean street width (meters)
        - StrOpen: Street openness ratio
        - StrWidthD: Street width deviation
        - StrLinM: Mean street linearity
        - StrLinS: Std street linearity
        - StrOrientM: Mean street orientation (degrees)
        - StrOrientS: Std street orientation
        - BldgStrAM: Mean building-street alignment (degrees)
        - BldgStrAS: Std building-street alignment

    Raises:
    -------
    ValueError
        If inputs have invalid CRS or geometries
    KeyError
        If required columns not found

    Examples:
    ---------
    >>> roads = gpd.read_file("roads.geojson")
    >>> cells = gpd.read_file("grid_cells.geojson")
    >>> result = calculate_street_network_metrics(roads, cells, n_jobs=4)
    >>> print(result[['grid_id', 'street_density', 'intersection_density']])
    """
    logger.info("Starting street network metrics calculation...")
    logger.info(f"Input: {len(roads):,} roads, {len(cells):,} cells")

    # Set default number of workers
    if n_jobs is None:
        n_jobs = cpu_count()
    logger.info(f"Using {n_jobs} parallel workers")

    # Define processing steps for progress bar
    total_steps = 12
    step_names = [
        "Validating inputs",
        "Projecting to target CRS",
        "Clipping roads to study area",
        "Building network graph",
        "Creating merged roads",
        "Calculating basic metrics (parallel)",
        "Aggregating density metrics",
        "Calculating advanced metrics (parallel)",
        "Calculating building-street alignment"
    ]

    # Create main progress bar
    with tqdm(total=total_steps, desc="  Street network", unit="step", position=0) as pbar:

        # ---- Step 1: Validate inputs ----
        pbar.set_postfix_str("Validating inputs")
        if cell_id_col not in cells.columns:
            raise KeyError(f"cells missing identifier column '{cell_id_col}'")
        pbar.update(1)

        # ---- Step 2: Project to target CRS ----
        pbar.set_postfix_str("Projecting to target CRS")
        roads = _project_crs(roads.copy(), target_crs)
        cells = _project_crs(cells.copy(), target_crs)
        buildings = _project_crs(buildings.copy(), target_crs) if buildings is not None else None

        # Ensure area column exists
        if area_col not in cells.columns:
            logger.info(f"  Area column '{area_col}' not found, calculating from geometry...")
            cells[area_col] = cells.geometry.area
        pbar.update(1)

        # ---- Step 3: Clip roads to study area ----
        pbar.set_postfix_str("Clipping roads")
        roads = _clip_roads_to_study_area(roads, cells)
        pbar.update(1)

        # ---- Step 4: Build network graph ----
        pbar.set_postfix_str("Building network graph")
        nodes_gdf, edges_gdf, G = _build_network_graph(roads)
        pbar.update(1)

        # ---- Step 5: Create merged roads (semantic "streets") ----
        pbar.set_postfix_str("Creating merged roads")
        logger.info("  Creating merged roads for semantic street metrics...")
        from shapely.ops import linemerge
        from shapely.geometry import LineString, MultiLineString

        degree_2_nodes = set()
        for node, degree in G.degree():
            if degree == 2:
                degree_2_nodes.add(node)

        logger.info(f"    Merging {len(edges_gdf):,} segments at {len(degree_2_nodes):,} degree-2 nodes...")

        # Use linemerge to combine connected segments
        all_edge_lines = list(edges_gdf.geometry)
        merged_geom = linemerge(all_edge_lines)

        if isinstance(merged_geom, LineString):
            merged_roads_geoms = [merged_geom]
        elif isinstance(merged_geom, MultiLineString):
            merged_roads_geoms = list(merged_geom.geoms)
        else:
            merged_roads_geoms = all_edge_lines

        merged_roads = gpd.GeoDataFrame(
            {'geometry': merged_roads_geoms},
            crs=edges_gdf.crs
        )

        logger.info(f"    Created {len(merged_roads):,} semantic streets from {len(edges_gdf):,} graph edges")
        pbar.update(1)

        # ---- Step 6: Calculate basic metrics in parallel (Group 1) ----
        pbar.set_postfix_str("Calculating basic metrics (parallel)")
        logger.info("  Calculating basic metrics in parallel...")

        # Use ThreadPoolExecutor for I/O-bound GIS operations
        length_by_cell = None
        segments_by_cell = None
        node_metrics = None
        plot_count = None

        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all independent tasks
            future_lengths = executor.submit(
                _calculate_street_lengths_per_cell,
                edges_gdf, cells, cell_id_col
            )
            future_nodes = executor.submit(
                _calculate_node_metrics_per_cell,
                nodes_gdf, cells, cell_id_col
            )
            future_plots = executor.submit(
                _calculate_plots_per_cell,
                edges_gdf, cells, cell_id_col, n_jobs
            )

            # Collect results as they complete
            for future in as_completed([future_lengths, future_nodes, future_plots]):
                if future == future_lengths:
                    length_by_cell, segments_by_cell = future.result()
                    logger.info("    Street lengths calculated")
                elif future == future_nodes:
                    node_metrics = future.result()
                    logger.info("    Node metrics calculated")
                elif future == future_plots:
                    plot_count = future.result()
                    logger.info("    Plot counts calculated")

        pbar.update(1)

        # ---- Step 6.5: Clip merged roads to cell boundaries ----
        # Street-level metrics (StreetDens, StrWidth, StrOpen, StrLin) are
        # computed from geometry clipped to each cell, so that a cell boundary
        # never inflates a metric with an out-of-cell street portion.
        logger.info("  Clipping merged roads to cell boundaries...")

        # Ensure spatial indices
        if not hasattr(merged_roads, '_sindex') or merged_roads._sindex is None:
            _ = merged_roads.sindex
        if not hasattr(cells, '_sindex') or cells._sindex is None:
            _ = cells.sindex

        # Prepare merged roads with unique IDs for overlay
        merged_roads_with_id = merged_roads.reset_index(drop=True).copy()
        merged_roads_with_id['road_id'] = range(len(merged_roads_with_id))

        # Overlay: clip merged roads to cell boundaries
        merged_roads_clipped = gpd.overlay(
            merged_roads_with_id,
            cells[[cell_id_col, 'geometry']],
            how='intersection',
            keep_geom_type=True
        )

        logger.info(f"    Clipped {len(merged_roads):,} roads into {len(merged_roads_clipped):,} cell-specific segments")

        # ---- Step 7: Aggregate density metrics ----
        pbar.set_postfix_str("Aggregating density metrics")
        logger.info("  Aggregating metrics to grid...")

        out = cells.copy()
        area_m2 = out[area_col].astype(float).replace(0, np.nan)  # Prevent division by zero

        # Street density (m/m²), from clipped merged roads so only in-cell length counts
        merged_roads_clipped['_length'] = merged_roads_clipped.geometry.length
        merged_length_by_cell = merged_roads_clipped.groupby(cell_id_col)['_length'].sum()
        out["StreetDens"] = merged_length_by_cell.reindex(out[cell_id_col]).fillna(0).values / area_m2.values

        # Segment density (count/m²)
        seg_cnt = segments_by_cell.reindex(out[cell_id_col]).fillna(0).astype(int)
        out["SegmentDen"] = seg_cnt.values / area_m2.values

        # Node metrics
        n_cnt = node_metrics["node_count"].reindex(out[cell_id_col]).fillna(0).astype(int).values
        out["NodeDens"] = n_cnt / area_m2.values
        out["StreetsPN"] = np.where(n_cnt > 0, seg_cnt.values / n_cnt, np.nan)

        # Connectivity metrics (DeadendDen, InterDens, Ratio3way, Ratio4way, CuldesacR)
        # are intentionally NOT calculated here. They are computed in the
        # connectivity module using a subgraph-based method that analyzes local
        # network topology correctly; calculating them here would overwrite
        # those more accurate results.
        logger.info("  Connectivity metrics (DeadendDen, InterDens, Ratio3way, Ratio4way, CuldesacR)")
        logger.info("      are calculated in the connectivity module using the subgraph method")

        # Plot density
        out["PlotDens"] = plot_count.reindex(out[cell_id_col]).fillna(0).values / area_m2.values

        # Building density (network-based)
        if buildings is not None:
            if not hasattr(buildings, '_sindex') or buildings._sindex is None:
                _ = buildings.sindex
            # Remove cell_id_col from buildings if it exists to avoid column name conflicts during sjoin
            buildings_temp = buildings.drop(columns=[cell_id_col], errors='ignore')
            buildings_cells_temp = gpd.sjoin(buildings_temp, cells[[cell_id_col, 'geometry']], predicate='within')
            buildings_cells_temp['_area'] = buildings_cells_temp.geometry.area
            building_area_by_cell = buildings_cells_temp.groupby(cell_id_col)['_area'].sum()
            out["BldgDensN"] = (building_area_by_cell.reindex(out[cell_id_col]).fillna(0).values / area_m2.values)
        else:
            out["BldgDensN"] = np.nan

        # Plot fragmentation.
        # The plot count is reduced by 1 to exclude the baseline plot formed
        # purely by the cell boundary, so a cell with no roads (only the one
        # boundary-enclosed plot) yields PlotFrag = 0 rather than a nonzero
        # fragmentation value caused only by the cell edge itself.
        plot_count_adjusted = plot_count.reindex(out[cell_id_col]).fillna(0).values - 1
        plot_count_adjusted = np.maximum(plot_count_adjusted, 0)  # Ensure non-negative
        plot_dens_adjusted = plot_count_adjusted / area_m2.values

        out["PlotFrag"] = np.where(out["BldgDensN"] > 0,
                                   plot_dens_adjusted / out["BldgDensN"],
                                   np.nan)
        pbar.update(1)

        # ---- Step 8: Calculate advanced metrics in parallel (Group 2) ----
        pbar.set_postfix_str("Calculating advanced metrics (parallel)")
        logger.info("  Calculating advanced metrics in parallel...")

        # Prepare clipped merged roads exploded for orientation (needed by multiple metrics)
        merged_roads_clipped_exploded = merged_roads_clipped.explode(index_parts=False).reset_index(drop=True)

        # Worker functions for parallel execution
        def calc_profile():
            """Calculate street profile metrics (width, openness)."""
            if buildings is None:
                return None, None, None
            try:
                # Sample only within cells by using the clipped roads
                profile = momepy.street_profile(
                    merged_roads_clipped,
                    buildings,
                    distance=10,
                    tick_length=50,
                    height=None
                )
                # Roads already carry cell_id_col, so a direct groupby suffices
                roads_with_profile = merged_roads_clipped.copy()
                roads_with_profile['width'] = profile['width']
                roads_with_profile['openness'] = profile['openness']
                roads_with_profile['width_deviation'] = profile['width_deviation']

                profile_agg = roads_with_profile.groupby(cell_id_col).agg({
                    'width': 'mean',
                    'openness': 'mean',
                    'width_deviation': 'mean'
                })
                return (
                    profile_agg['width'].reindex(out[cell_id_col]).values,
                    profile_agg['openness'].reindex(out[cell_id_col]).values,
                    profile_agg['width_deviation'].reindex(out[cell_id_col]).values
                )
            except Exception as e:
                logger.warning(f"  Street profile calculation failed: {str(e)}")
                return None, None, None

        def calc_linearity():
            """Calculate street linearity metrics."""
            from shapely.geometry import LineString, MultiLineString

            def calculate_linearity_safe(geom):
                if geom.is_empty or geom.length == 0:
                    return np.nan

                if isinstance(geom, LineString):
                    coords = list(geom.coords)
                    if len(coords) < 2:
                        return np.nan
                    start = np.array(coords[0])
                    end = np.array(coords[-1])
                    euclidean = np.linalg.norm(end - start)
                    return euclidean / geom.length if geom.length > 0 else np.nan

                elif isinstance(geom, MultiLineString):
                    geoms_list = list(geom.geoms)
                    if len(geoms_list) == 0:
                        return np.nan

                    continuous_segments = [[geoms_list[0]]]
                    for i in range(len(geoms_list) - 1):
                        end_point = np.array(list(geoms_list[i].coords)[-1])
                        start_point = np.array(list(geoms_list[i+1].coords)[0])
                        gap = np.linalg.norm(end_point - start_point)
                        if gap > 1.0:
                            continuous_segments.append([geoms_list[i+1]])
                        else:
                            continuous_segments[-1].append(geoms_list[i+1])

                    total_length = 0
                    weighted_linearity = 0
                    for segment_parts in continuous_segments:
                        segment_length = sum(part.length for part in segment_parts)
                        if segment_length == 0:
                            continue
                        start = np.array(list(segment_parts[0].coords)[0])
                        end = np.array(list(segment_parts[-1].coords)[-1])
                        euclidean = np.linalg.norm(end - start)
                        linearity = min(euclidean / segment_length, 1.0)
                        weighted_linearity += linearity * segment_length
                        total_length += segment_length

                    return weighted_linearity / total_length if total_length > 0 else np.nan
                return np.nan

            # Use clipped roads so linearity is only calculated within cells
            linearity_values = merged_roads_clipped.geometry.apply(calculate_linearity_safe)
            roads_with_linearity = merged_roads_clipped.copy()
            roads_with_linearity['linearity'] = linearity_values
            # Roads already carry cell_id_col, so a direct groupby suffices
            linearity_agg = roads_with_linearity.groupby(cell_id_col)['linearity'].agg(['mean', 'std'])
            return (
                linearity_agg['mean'].reindex(out[cell_id_col]).values,
                linearity_agg['std'].reindex(out[cell_id_col]).values
            )

        def calc_orientation():
            """Calculate street orientation metrics."""
            from scipy.stats import circmean, circstd

            # Use clipped exploded roads so orientation is only calculated within cells
            street_orientation = momepy.orientation(merged_roads_clipped_exploded)
            roads_with_orient = merged_roads_clipped_exploded.copy()
            roads_with_orient['orientation'] = street_orientation

            def circular_stats_orient(x):
                if len(x) == 0 or x.isna().all():
                    return pd.Series({'circular_mean': np.nan, 'circular_std': np.nan})
                return pd.Series({
                    'circular_mean': circmean(x.dropna(), high=180, low=0),
                    'circular_std': circstd(x.dropna(), high=180, low=0)
                })

            # Roads already carry cell_id_col, so a direct groupby suffices
            orient_agg = roads_with_orient.groupby(cell_id_col)['orientation'].apply(circular_stats_orient).unstack()
            return (
                orient_agg['circular_mean'].reindex(out[cell_id_col]).values,
                orient_agg['circular_std'].reindex(out[cell_id_col]).values,
                street_orientation  # Return this for building-street alignment
            )

        # Execute Group 2 metrics in parallel
        profile_result = None
        linearity_result = None
        orientation_result = None
        street_orientation = None

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_profile = executor.submit(calc_profile)
            future_linearity = executor.submit(calc_linearity)
            future_orientation = executor.submit(calc_orientation)

            for future in as_completed([future_profile, future_linearity, future_orientation]):
                if future == future_profile:
                    width, openness, width_dev = future.result()
                    out['StrWidth'] = width if width is not None else np.nan
                    out['StrOpen'] = openness if openness is not None else np.nan
                    out['StrWidthD'] = width_dev if width_dev is not None else np.nan
                    logger.info("    Street profile calculated")
                elif future == future_linearity:
                    lin_mean, lin_std = future.result()
                    out['StrLinM'] = lin_mean
                    out['StrLinS'] = lin_std
                    logger.info("    Street linearity calculated")
                elif future == future_orientation:
                    orient_mean, orient_std, street_orientation = future.result()
                    out['StrOrientM'] = orient_mean
                    out['StrOrientS'] = orient_std
                    logger.info("    Street orientation calculated")

        pbar.update(1)

        # ---- Step 9: Calculate building-street alignment (depends on orientation) ----
        pbar.set_postfix_str("Building-street alignment")
        logger.info("  Calculating building-street alignment...")
        if buildings is not None and 'Orientation' in buildings.columns and street_orientation is not None:
            buildings_with_street = buildings.copy()
            # Use clipped exploded roads for nearest street matching
            buildings_with_street['nearest_street_idx'] = momepy.get_nearest_street(buildings, merged_roads_clipped_exploded)
            building_orient = buildings['Orientation']
            # Street orientation is from the clipped roads
            street_orient_series = pd.Series(street_orientation.values, index=merged_roads_clipped_exploded.index)
            alignment = momepy.street_alignment(
                building_orient,
                street_orient_series,
                buildings_with_street['nearest_street_idx']
            )
            buildings_with_street['street_alignment'] = alignment
            # Remove cell_id_col from buildings_with_street if it exists to avoid column name conflicts during sjoin
            buildings_with_street_temp = buildings_with_street.drop(columns=[cell_id_col], errors='ignore')
            buildings_cells = gpd.sjoin(buildings_with_street_temp, cells[[cell_id_col, 'geometry']], predicate='within')
            alignment_agg = buildings_cells.groupby(cell_id_col)['street_alignment'].agg(['mean', 'std'])
            out['BldgStrAM'] = alignment_agg['mean'].reindex(out[cell_id_col]).values
            out['BldgStrAS'] = alignment_agg['std'].reindex(out[cell_id_col]).values
        else:
            if buildings is None:
                logger.warning("  Buildings not provided, skipping building-street alignment")
            else:
                logger.warning("  Building orientation not found, skipping building-street alignment")
            out['BldgStrAM'] = np.nan
            out['BldgStrAS'] = np.nan
        pbar.update(1)

    logger.info(f"Street network metrics calculation completed for {len(out):,} cells")

    return out


def calculate_street_network_metrics_with_graph(
    roads: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    buildings: Optional[gpd.GeoDataFrame] = None,
    cell_id_col: str = "grid_id",
    area_col: str = "Area_m2",
    target_crs: str = "ESRI:102022",
    n_jobs: int = None
) -> tuple:
    """
    Calculate street network metrics and return graph objects for connectivity analysis.

    This function is functionally equivalent to calculate_street_network_metrics()
    but also returns the network graph objects needed by the connectivity metrics
    module, and does not clip merged roads to cell boundaries before computing
    street-level metrics (it uses spatial-join based aggregation instead).

    Parameters:
    -----------
    Same as calculate_street_network_metrics()

    Returns:
    --------
    tuple
        (cells_with_metrics, nodes_gdf, edges_gdf, graph)
        - cells_with_metrics: GeoDataFrame with street network metrics
        - nodes_gdf: GeoDataFrame of network nodes
        - edges_gdf: GeoDataFrame of network edges (with 'length' column)
        - graph: nx.MultiGraph network graph object

    Notes:
    ------
    - The graph is the cleaned network (after false-node removal)
    - The edges_gdf includes a 'length' column required for connectivity metrics

    Example:
    --------
    >>> cells, nodes, edges, G = calculate_street_network_metrics_with_graph(
    ...     roads, cells, buildings, n_jobs=4
    ... )
    >>> # Now pass nodes, edges, G to connectivity_metrics.calculate_connectivity_metrics()
    """
    logger.info("Starting street network metrics calculation WITH GRAPH OBJECTS...")
    logger.info(f"Input: {len(roads):,} roads, {len(cells):,} cells")

    # Set default number of workers
    if n_jobs is None:
        n_jobs = cpu_count()
    logger.info(f"Using {n_jobs} parallel workers")

    # Define processing steps for progress bar
    total_steps = 12
    step_names = [
        "Validating inputs",
        "Projecting to target CRS",
        "Clipping roads to study area",
        "Building network graph",
        "Creating merged roads",
        "Calculating basic metrics",
        "Calculating plot density",
        "Calculating density metrics",
        "Calculating advanced metrics (parallel)",
        "Street profile",
        "Street linearity",
        "Building-street alignment"
    ]

    with tqdm(total=total_steps, desc="Street Network Metrics", position=0, leave=True) as pbar:
        # ---- Step 1: Input validation ----
        pbar.set_postfix_str(step_names[0])
        logger.info(f"  {step_names[0]}...")

        if roads is None or len(roads) == 0:
            raise ValueError("Roads GeoDataFrame is empty or None")
        if cells is None or len(cells) == 0:
            raise ValueError("Cells GeoDataFrame is empty or None")
        if roads.crs is None:
            raise ValueError("Roads GeoDataFrame does not have CRS")
        if cells.crs is None:
            raise ValueError("Cells GeoDataFrame does not have CRS")

        pbar.update(1)

        # ---- Step 2: Project to target CRS ----
        pbar.set_postfix_str(step_names[1])
        logger.info(f"  {step_names[1]}...")

        roads = _project_crs(roads, target_crs)
        cells = _project_crs(cells, target_crs)
        if buildings is not None:
            buildings = _project_crs(buildings, target_crs)

        pbar.update(1)

        # ---- Step 3: Clip roads to study area ----
        pbar.set_postfix_str(step_names[2])
        roads_clipped = _clip_roads_to_study_area(roads, cells)
        pbar.update(1)

        # ---- Step 4: Build network graph ----
        pbar.set_postfix_str(step_names[3])
        nodes_gdf, edges_gdf, graph = _build_network_graph(roads_clipped)

        # Graph is stored here so it can be returned for connectivity analysis
        logger.info(f"  Graph built: {len(nodes_gdf):,} nodes, {len(edges_gdf):,} edges")

        pbar.update(1)

        # ---- Step 5: Create merged roads for linearity ----
        pbar.set_postfix_str(step_names[4])
        logger.info("  Creating merged roads for semantic street metrics...")
        from shapely.ops import linemerge
        from shapely.geometry import LineString, MultiLineString

        degree_2_nodes = set()
        for node, degree in graph.degree():
            if degree == 2:
                degree_2_nodes.add(node)

        logger.info(f"    Merging {len(edges_gdf):,} segments at {len(degree_2_nodes):,} degree-2 nodes...")

        # Use linemerge to combine connected segments
        all_edge_lines = list(edges_gdf.geometry)
        merged_geom = linemerge(all_edge_lines)

        if isinstance(merged_geom, LineString):
            merged_roads_geoms = [merged_geom]
        elif isinstance(merged_geom, MultiLineString):
            merged_roads_geoms = list(merged_geom.geoms)
        else:
            merged_roads_geoms = all_edge_lines

        merged_roads = gpd.GeoDataFrame(
            {'geometry': merged_roads_geoms},
            crs=edges_gdf.crs
        )

        logger.info(f"    Created {len(merged_roads):,} semantic streets from {len(edges_gdf):,} graph edges")
        pbar.update(1)

        # ---- Step 6: Calculate basic metrics ----
        pbar.set_postfix_str(step_names[5])
        logger.info("  Calculating basic metrics...")

        # Initialize output dataframe
        out = cells.copy()

        # Calculate area if not present
        if area_col not in out.columns:
            logger.info(f"  '{area_col}' column not found, calculating from geometry...")
            out[area_col] = out.geometry.area
        area_m2 = out[area_col]

        # Calculate street lengths and segments per cell
        length_by_cell, segments_by_cell = _calculate_street_lengths_per_cell(
            edges_gdf, cells, cell_id_col
        )

        # Calculate node metrics per cell
        node_metrics = _calculate_node_metrics_per_cell(
            nodes_gdf, cells, cell_id_col
        )

        pbar.update(1)

        # ---- Step 7: Calculate plot density (parallel processing) ----
        pbar.set_postfix_str(step_names[6])
        logger.info("  Calculating plot density (parallel processing)...")

        plot_counts = _calculate_plots_per_cell(edges_gdf, cells, cell_id_col, n_jobs)
        pbar.update(1)

        # ---- Step 8: Compute density metrics ----
        pbar.set_postfix_str(step_names[7])
        logger.info("  Computing density metrics...")

        # Street density
        out["StreetDens"] = length_by_cell.reindex(out[cell_id_col]).fillna(0).values / area_m2.values

        # Node metrics
        n_cnt = node_metrics["node_count"].reindex(out[cell_id_col]).fillna(0).astype(int).values
        out["NodeDens"] = n_cnt / area_m2.values
        out["StreetsPN"] = np.where(n_cnt > 0, segments_by_cell.reindex(out[cell_id_col]).fillna(0).values / n_cnt, np.nan)

        # Connectivity metrics (DeadendDen, InterDens, Ratio3way, Ratio4way, CuldesacR)
        # are intentionally NOT calculated here. They are computed in the
        # connectivity module using a subgraph-based method that analyzes local
        # network topology correctly; calculating them here would overwrite
        # those more accurate results.
        logger.info("  Connectivity metrics (DeadendDen, InterDens, Ratio3way, Ratio4way, CuldesacR)")
        logger.info("      are calculated in the connectivity module using the subgraph method")

        # Segment density
        out["SegmentDen"] = segments_by_cell.reindex(out[cell_id_col]).fillna(0).values / area_m2.values

        # Plot density
        out["PlotDens"] = plot_counts.reindex(out[cell_id_col]).fillna(0).values / area_m2.values

        # Building density (network-based)
        if buildings is not None:
            if not hasattr(buildings, '_sindex') or buildings._sindex is None:
                _ = buildings.sindex
            # Remove cell_id_col from buildings if it exists to avoid column name conflicts during sjoin
            buildings_temp = buildings.drop(columns=[cell_id_col], errors='ignore')
            buildings_cells_temp = gpd.sjoin(buildings_temp, cells[[cell_id_col, 'geometry']], predicate='within')
            buildings_cells_temp['_area'] = buildings_cells_temp.geometry.area
            building_area_by_cell = buildings_cells_temp.groupby(cell_id_col)['_area'].sum()
            out["BldgDensN"] = (building_area_by_cell.reindex(out[cell_id_col]).fillna(0).values / area_m2.values)
        else:
            out["BldgDensN"] = np.nan

        # Plot fragmentation.
        # The plot count is reduced by 1 to exclude the baseline plot formed
        # purely by the cell boundary, so a cell with no roads (only the one
        # boundary-enclosed plot) yields PlotFrag = 0 rather than a nonzero
        # fragmentation value caused only by the cell edge itself.
        plot_count_adjusted = plot_counts.reindex(out[cell_id_col]).fillna(0).values - 1
        plot_count_adjusted = np.maximum(plot_count_adjusted, 0)  # Ensure non-negative
        plot_dens_adjusted = plot_count_adjusted / area_m2.values

        out["PlotFrag"] = np.where(out["BldgDensN"] > 0,
                                   plot_dens_adjusted / out["BldgDensN"],
                                   np.nan)
        pbar.update(1)

        # ---- Step 9: Calculate advanced metrics in parallel (Group 2) ----
        pbar.set_postfix_str(step_names[8])
        logger.info("  Calculating advanced metrics in parallel...")

        # Prepare merged roads exploded for orientation (needed by multiple metrics)
        merged_roads_exploded = merged_roads.explode(index_parts=False).reset_index(drop=True)

        # Street orientation is calculated once and shared by calc_orientation()
        # and calc_building_alignment() to avoid redundant computation.
        logger.info("  Pre-calculating street orientation (shared computation)...")
        street_orientation = momepy.orientation(merged_roads_exploded)
        streets_with_orientation = merged_roads_exploded.copy()
        streets_with_orientation['orientation'] = street_orientation

        # Worker functions for parallel execution
        def calc_profile():
            """Calculate street profile metrics (width, openness)."""
            if buildings is None:
                return None, None, None
            try:
                profile = momepy.street_profile(
                    merged_roads,
                    buildings,
                    distance=10,
                    tick_length=50,
                    height=None
                )
                merged_roads_with_profile = merged_roads.copy()
                merged_roads_with_profile['width'] = profile['width']
                merged_roads_with_profile['openness'] = profile['openness']
                merged_roads_with_profile['width_deviation'] = profile['width_deviation']

                roads_cells = gpd.sjoin(merged_roads_with_profile, cells[[cell_id_col, 'geometry']], predicate='intersects')
                profile_agg = roads_cells.groupby(cell_id_col).agg({
                    'width': 'mean',
                    'openness': 'mean',
                    'width_deviation': 'mean'
                })
                return (
                    profile_agg['width'].reindex(out[cell_id_col]).values,
                    profile_agg['openness'].reindex(out[cell_id_col]).values,
                    profile_agg['width_deviation'].reindex(out[cell_id_col]).values
                )
            except Exception as e:
                logger.warning(f"  Street profile calculation failed: {str(e)}")
                return None, None, None

        def calc_linearity():
            """Calculate street linearity metrics."""
            try:
                logger.info(f"  [Linearity] Calculating linearity for {len(merged_roads_exploded):,} segments...")
                linearity = momepy.linearity(merged_roads_exploded)
                logger.info("  [Linearity] Linearity calculated, aggregating...")
                merged_with_linearity = merged_roads_exploded.copy()
                merged_with_linearity['linearity'] = linearity

                roads_cells = gpd.sjoin(merged_with_linearity, cells[[cell_id_col, 'geometry']], predicate='intersects')
                linearity_agg = roads_cells.groupby(cell_id_col)['linearity'].agg(['mean', 'std'])
                logger.info("  [Linearity] Linearity metrics complete!")
                return (
                    linearity_agg['mean'].reindex(out[cell_id_col]).values,
                    linearity_agg['std'].reindex(out[cell_id_col]).values
                )
            except Exception as e:
                logger.warning(f"  Street linearity calculation failed: {str(e)}")
                return None, None

        def calc_orientation():
            """Calculate street orientation metrics."""
            try:
                logger.info(f"  [Orientation] Aggregating orientation to {len(cells):,} cells...")
                # Reuse pre-calculated orientation instead of recalculating
                roads_cells = gpd.sjoin(streets_with_orientation, cells[[cell_id_col, 'geometry']], predicate='intersects')
                logger.info(f"  [Orientation] Spatial join complete, calculating statistics...")

                # Calculate circular mean and std
                from scipy import stats

                def circular_mean(angles_deg):
                    if len(angles_deg) == 0:
                        return np.nan
                    angles_rad = np.deg2rad(angles_deg)
                    return np.rad2deg(stats.circmean(angles_rad, high=np.pi, low=-np.pi))

                def circular_std(angles_deg):
                    if len(angles_deg) == 0:
                        return np.nan
                    angles_rad = np.deg2rad(angles_deg)
                    return np.rad2deg(stats.circstd(angles_rad, high=np.pi, low=-np.pi))

                orientation_agg = roads_cells.groupby(cell_id_col)['orientation'].agg([circular_mean, circular_std])
                logger.info("  [Orientation] Orientation metrics complete!")
                return (
                    orientation_agg['circular_mean'].reindex(out[cell_id_col]).values,
                    orientation_agg['circular_std'].reindex(out[cell_id_col]).values
                )
            except Exception as e:
                logger.warning(f"  Street orientation calculation failed: {str(e)}")
                return None, None

        def calc_building_alignment():
            """Calculate building-street alignment."""
            if buildings is None:
                return None, None
            try:
                logger.info(f"  [Alignment] Starting with {len(buildings):,} buildings...")

                # Use existing building orientation if available (from morphology metrics)
                if 'Orientation' in buildings.columns:
                    building_orientation = buildings['Orientation']
                    logger.info("  [Alignment] Using existing building orientation from morphology")
                else:
                    logger.info("  [Alignment] Calculating building orientation...")
                    building_orientation = momepy.Orientation(buildings).series
                    logger.info("  [Alignment] Building orientation calculated")

                buildings_with_orientation = buildings.copy()
                buildings_with_orientation['building_orientation'] = building_orientation

                # Reuse pre-calculated street orientation (streets_with_orientation
                # already has the 'orientation' column from the earlier pre-calculation)

                # If buildings already carry cell_id_col, the expensive spatial
                # join can be skipped entirely.
                if cell_id_col in buildings_with_orientation.columns:
                    logger.info(f"  [Alignment] Buildings already have {cell_id_col}, skipping spatial join!")
                    buildings_cells = buildings_with_orientation[[cell_id_col, 'building_orientation', 'geometry']].copy()
                    buildings_cells = buildings_cells[buildings_cells[cell_id_col].notna()]
                    logger.info(f"  [Alignment] Using {len(buildings_cells):,} buildings with valid {cell_id_col}")
                else:
                    # Calculate alignment - drop cell_id_col from buildings if exists to avoid conflicts
                    buildings_temp = buildings_with_orientation.drop(columns=[cell_id_col], errors='ignore')

                    # Ensure spatial index exists before sjoin
                    if not hasattr(buildings_temp, '_sindex') or buildings_temp._sindex is None:
                        logger.info(f"  [Alignment] Building spatial index for {len(buildings_temp):,} buildings...")
                        _ = buildings_temp.sindex
                    if not hasattr(cells, '_sindex') or cells._sindex is None:
                        _ = cells.sindex

                    logger.info(f"  [Alignment] Spatial join: {len(buildings_temp):,} buildings -> cells (this may take a while)...")
                    buildings_cells = gpd.sjoin(buildings_temp, cells[[cell_id_col, 'geometry']], predicate='within')
                    logger.info(f"  [Alignment] Building-cell join complete: {len(buildings_cells):,} matches")

                logger.info(f"  [Alignment] Spatial join: {len(streets_with_orientation):,} streets -> cells...")
                streets_cells = gpd.sjoin(streets_with_orientation, cells[[cell_id_col, 'geometry']], predicate='intersects')
                logger.info(f"  [Alignment] Street-cell join complete: {len(streets_cells):,} matches")

                # Group by cell and calculate mean alignment
                from scipy import stats

                logger.info("  [Alignment] Calculating circular means by cell...")
                building_mean = buildings_cells.groupby(cell_id_col)['building_orientation'].apply(
                    lambda x: np.rad2deg(stats.circmean(np.deg2rad(x), high=np.pi, low=-np.pi))
                )
                # Use 'orientation' column name (from pre-calculation)
                street_mean = streets_cells.groupby(cell_id_col)['orientation'].apply(
                    lambda x: np.rad2deg(stats.circmean(np.deg2rad(x), high=np.pi, low=-np.pi))
                )
                logger.info("  [Alignment] Circular means calculated")

                alignment_df = pd.DataFrame({
                    'building_mean': building_mean,
                    'street_mean': street_mean
                })

                # Vectorized calculation instead of apply (much faster)
                logger.info("  [Alignment] Calculating alignment differences...")
                diff = (alignment_df['building_mean'] - alignment_df['street_mean']).abs()
                alignment_mean = pd.Series(np.minimum(diff, 180 - diff), index=alignment_df.index)

                # Calculate std within each cell
                logger.info("  [Alignment] Calculating circular standard deviations...")
                alignment_std = buildings_cells.groupby(cell_id_col)['building_orientation'].apply(
                    lambda x: np.rad2deg(stats.circstd(np.deg2rad(x), high=np.pi, low=-np.pi))
                )
                logger.info("  [Alignment] Building-street alignment complete!")

                return (
                    alignment_mean.reindex(out[cell_id_col]).values,
                    alignment_std.reindex(out[cell_id_col]).values
                )
            except Exception as e:
                logger.warning(f"  Building-street alignment calculation failed: {str(e)}")
                return None, None

        # Run workers in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_profile = executor.submit(calc_profile)
            future_linearity = executor.submit(calc_linearity)
            future_orientation = executor.submit(calc_orientation)
            future_alignment = executor.submit(calc_building_alignment)

            # Collect results
            width, openness, width_dev = future_profile.result()
            linearity_mean, linearity_std = future_linearity.result()
            orientation_mean, orientation_std = future_orientation.result()
            alignment_mean, alignment_std = future_alignment.result()

        # Assign to output
        out["StrWidth"] = width if width is not None else np.nan
        out["StrOpen"] = openness if openness is not None else np.nan
        out["StrWidthD"] = width_dev if width_dev is not None else np.nan
        out["StrLinM"] = linearity_mean if linearity_mean is not None else np.nan
        out["StrLinS"] = linearity_std if linearity_std is not None else np.nan
        out["StrOrientM"] = orientation_mean if orientation_mean is not None else np.nan
        out["StrOrientS"] = orientation_std if orientation_std is not None else np.nan
        out["BldgStrAM"] = alignment_mean if alignment_mean is not None else np.nan
        out["BldgStrAS"] = alignment_std if alignment_std is not None else np.nan

        pbar.update(4)  # Update for all 4 parallel tasks

    logger.info(f"Street network metrics calculation completed for {len(out):,} cells")
    logger.info(f"  Returning graph objects: {len(nodes_gdf):,} nodes, {len(edges_gdf):,} edges")

    # Ensure edges_gdf has 'length' column for connectivity metrics
    if 'length_m' in edges_gdf.columns and 'length' not in edges_gdf.columns:
        edges_gdf['length'] = edges_gdf['length_m']

    return out, nodes_gdf, edges_gdf, graph


# ============================================================================
# COLUMN NAME MAPPING GENERATION
# ============================================================================

def get_street_network_column_mappings() -> list:
    """
    Get column name mappings for street network metrics.

    Returns list of tuples for integration into column_name_mapping.csv.
    Format: (Short_Name, Full_Name, Data_Source, Momepy_Category, Statistic_Type, Description)

    Categories:
    - Intensity: density and ratio metrics
    - Dimension: width, openness metrics
    - Shape: linearity metrics
    - Spatial Distribution: orientation and alignment metrics
    - Diversity: std metrics

    Returns:
    --------
    list
        List of tuples with column mappings
    """
    mappings = [
        # ---- Intensity Metrics ----
        ('StreetDens', 'Street Density', 'Momepy', 'Intensity', 'Density',
         'Total street length per unit area (m/m²)'),

        ('NodeDens', 'Node Density', 'Momepy', 'Intensity', 'Density',
         'Number of network nodes per unit area (count/m²)'),

        ('StreetsPN', 'Streets Per Node', 'Momepy', 'Intensity', 'Ratio',
         'Average number of street segments per node'),

        ('SegmentDen', 'Segment Density', 'Momepy', 'Intensity', 'Density',
         'Number of street segments per unit area (count/m²)'),

        ('DeadendDen', 'Dead End Density', 'Momepy', 'Intensity', 'Density',
         'Number of dead ends (degree-1 nodes) per unit area (count/m²)'),

        ('InterDens', 'Intersection Density', 'Momepy', 'Intensity', 'Density',
         'Number of intersections (degree ≥ 3 nodes) per unit area (count/m²)'),

        ('Ratio3way', '3-way Intersection Ratio', 'Momepy', 'Connectivity', 'Ratio',
         'Ratio of 3-way intersections to total intersections'),

        ('Ratio4way', '4-way Intersection Ratio', 'Momepy', 'Connectivity', 'Ratio',
         'Ratio of 4-way intersections to total intersections'),

        ('CuldesacR', 'Cul-de-sac Ratio', 'Momepy', 'Connectivity', 'Ratio',
         'Ratio of dead ends to intersections'),

        ('PlotDens', 'Plot Density', 'Momepy', 'Intensity', 'Density',
         'Number of plots (polygonized areas) per unit area (count/m²)'),

        # ---- Dimension Metrics ----
        ('StrWidth', 'Street Width', 'Momepy', 'Dimension', 'Mean',
         'Mean street width measured perpendicular to street centerline (meters)'),

        ('StrOpen', 'Street Openness', 'Momepy', 'Dimension', 'Mean',
         'Mean street openness ratio (proportion of ticks not reaching buildings)'),

        ('StrWidthD', 'Street Width Deviation', 'Momepy', 'Diversity', 'Std',
         'Mean standard deviation of street width (diversity indicator)'),

        # ---- Shape Metrics ----
        ('StrLinM', 'Street Linearity Mean', 'Momepy', 'Shape', 'Mean',
         'Mean street linearity (ratio of euclidean distance to actual length)'),

        ('StrLinS', 'Street Linearity Std', 'Momepy', 'Diversity', 'Std',
         'Standard deviation of street linearity (diversity)'),

        # ---- Spatial Distribution Metrics ----
        ('StrOrientM', 'Street Orientation Mean', 'Momepy', 'Spatial Distribution', 'Mean',
         'Mean street orientation angle (degrees, computed using circular mean)'),

        ('StrOrientS', 'Street Orientation Std', 'Momepy', 'Diversity', 'Std',
         'Standard deviation of street orientation (diversity, computed using circular std)'),

        ('BldgStrAM', 'Building-Street Alignment Mean', 'Momepy', 'Spatial Distribution', 'Mean',
         'Mean deviation of building orientation from nearest street orientation (degrees)'),

        ('BldgStrAS', 'Building-Street Alignment Std', 'Momepy', 'Diversity', 'Std',
         'Standard deviation of building-street alignment (diversity)'),
    ]

    return mappings
