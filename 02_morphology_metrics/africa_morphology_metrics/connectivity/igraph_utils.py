"""
NetworkX <-> igraph conversion utilities.

igraph is a high-performance graph library with a C implementation, used by
the igraph-backed connectivity implementations for centrality calculations
that are otherwise a bottleneck in pure Python NetworkX.
"""

import networkx as nx
import igraph as ig
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


def nx_to_igraph(G: nx.MultiGraph) -> Tuple[ig.Graph, Dict]:
    """
    Convert a NetworkX MultiGraph to an igraph Graph.

    NetworkX nodes are (x, y) coordinate tuples in the street network graphs
    used by this pipeline. igraph uses integer vertex indices (0-based), so
    this builds the index mapping needed to translate between the two.

    Parameters
    ----------
    G : nx.MultiGraph
        NetworkX graph with 'mm_len' edge weights.

    Returns
    -------
    g : ig.Graph
        igraph Graph object.
    mappings : dict
        - 'node_to_vertex': {nx_node_id: ig_vertex_index}
        - 'vertex_to_node': {ig_vertex_index: nx_node_id}
        - 'edge_weights': list of edge weights (mm_len)
        - 'node_coords': list of (x, y) tuples for each vertex

    Example
    -------
    >>> G = nx.MultiGraph()
    >>> G.add_node((0.0, 0.0), x=0.0, y=0.0)
    >>> G.add_node((1.0, 0.0), x=1.0, y=0.0)
    >>> G.add_edge((0.0, 0.0), (1.0, 0.0), mm_len=100.0)
    >>> g, mappings = nx_to_igraph(G)
    >>> print(f"Vertices: {g.vcount()}, Edges: {g.ecount()}")
    Vertices: 2, Edges: 1
    """
    # Create node-to-vertex mapping
    node_to_vertex = {}
    vertex_to_node = {}
    node_coords = []

    for i, node in enumerate(G.nodes()):
        node_to_vertex[node] = i
        vertex_to_node[i] = node

        # Extract coordinates
        if isinstance(node, tuple) and len(node) == 2:
            node_coords.append(node)
        else:
            # Fallback: get from node attributes
            x = G.nodes[node].get('x', 0.0)
            y = G.nodes[node].get('y', 0.0)
            node_coords.append((x, y))

    # Create igraph graph (undirected, since street networks are bidirectional)
    n_vertices = len(G.nodes())
    g = ig.Graph(n=n_vertices, directed=False)

    # Add edges with weights
    edge_list = []
    edge_weights = []

    for u, v, key, data in G.edges(keys=True, data=True):
        u_idx = node_to_vertex[u]
        v_idx = node_to_vertex[v]
        edge_list.append((u_idx, v_idx))

        # Get edge weight (mm_len in meters)
        weight = data.get('mm_len', 1.0)
        edge_weights.append(weight)

    # Add all edges at once (faster than one-by-one)
    g.add_edges(edge_list)

    # Store edge weights as edge attribute
    g.es['weight'] = edge_weights

    # Store node coordinates as vertex attributes
    g.vs['x'] = [coords[0] for coords in node_coords]
    g.vs['y'] = [coords[1] for coords in node_coords]

    mappings = {
        'node_to_vertex': node_to_vertex,
        'vertex_to_node': vertex_to_node,
        'edge_weights': edge_weights,
        'node_coords': node_coords
    }

    return g, mappings


def igraph_to_nx(g: ig.Graph, mappings: Dict) -> nx.MultiGraph:
    """
    Convert an igraph Graph back to a NetworkX MultiGraph.

    Used for verification and debugging (round-trip conversion checks).

    Parameters
    ----------
    g : ig.Graph
        igraph graph object.
    mappings : dict
        Mappings produced by `nx_to_igraph`.

    Returns
    -------
    G : nx.MultiGraph
        NetworkX graph.
    """
    G = nx.MultiGraph()

    # Add nodes
    vertex_to_node = mappings['vertex_to_node']
    node_coords = mappings['node_coords']

    for v_idx in range(g.vcount()):
        node_id = vertex_to_node[v_idx]
        x, y = node_coords[v_idx]
        G.add_node(node_id, x=x, y=y)

    # Add edges
    edge_weights = g.es['weight']

    for i, edge in enumerate(g.es):
        u_idx = edge.source
        v_idx = edge.target
        u_node = vertex_to_node[u_idx]
        v_node = vertex_to_node[v_idx]
        weight = edge_weights[i]

        G.add_edge(u_node, v_node, mm_len=weight)

    return G


def get_target_vertices(
    target_node_ids: List,
    node_to_vertex: Dict
) -> List[int]:
    """
    Convert a list of NetworkX node IDs to igraph vertex indices.

    Parameters
    ----------
    target_node_ids : list
        NetworkX node IDs (typically (x, y) tuples).
    node_to_vertex : dict
        Mapping from nx node ID to ig vertex index.

    Returns
    -------
    target_vertices : list of int
        igraph vertex indices.
    """
    return [node_to_vertex[nid] for nid in target_node_ids if nid in node_to_vertex]


def extract_vertex_values(
    result_array: List[float],
    target_vertices: List[int],
    vertex_to_node: Dict
) -> Dict:
    """
    Extract values for target vertices and return as {node_id: value}.

    Parameters
    ----------
    result_array : list of float
        Array of values (one per vertex).
    target_vertices : list of int
        Indices of target vertices.
    vertex_to_node : dict
        Mapping from vertex index to node ID.

    Returns
    -------
    node_values : dict
        {node_id: value} for target nodes.
    """
    node_values = {}
    for v_idx in target_vertices:
        node_id = vertex_to_node[v_idx]
        node_values[node_id] = result_array[v_idx]

    return node_values


def _test_conversion():
    """Round-trip conversion self-test (nx -> igraph -> nx)."""
    G = nx.MultiGraph()
    G.add_node((0.0, 0.0), x=0.0, y=0.0)
    G.add_node((1.0, 0.0), x=1.0, y=0.0)
    G.add_node((0.0, 1.0), x=0.0, y=1.0)
    G.add_edge((0.0, 0.0), (1.0, 0.0), mm_len=100.0)
    G.add_edge((0.0, 0.0), (0.0, 1.0), mm_len=150.0)

    print("Original NetworkX graph:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    g, mappings = nx_to_igraph(G)

    print("\nConverted to igraph:")
    print(f"  Vertices: {g.vcount()}")
    print(f"  Edges: {g.ecount()}")
    print(f"  Edge weights: {g.es['weight']}")

    G_restored = igraph_to_nx(g, mappings)

    print("\nConverted back to NetworkX:")
    print(f"  Nodes: {G_restored.number_of_nodes()}")
    print(f"  Edges: {G_restored.number_of_edges()}")

    assert G.number_of_nodes() == G_restored.number_of_nodes(), "Node count mismatch!"
    assert G.number_of_edges() == G_restored.number_of_edges(), "Edge count mismatch!"

    print("\nConversion test passed.")

    return g, mappings


if __name__ == "__main__":
    _test_conversion()
