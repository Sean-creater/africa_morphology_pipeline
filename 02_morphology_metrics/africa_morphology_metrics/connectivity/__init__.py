"""
Street network connectivity metrics.

Three implementations of `calculate_connectivity_metrics` are available,
in order of preference:

1. `connectivity_igraph_primary` (primary) - igraph-accelerated, no
   neighbor-cell buffer, adds clustering coefficient, COINS street
   continuity, and basic intersection-type metrics.
2. `connectivity_igraph_buffered` - igraph-accelerated, 6-neighbor cell buffer.
3. `connectivity_networkx` - pure NetworkX/momepy, no igraph
   dependency; used only when python-igraph is not installed.

Selection: attempt to import `igraph` itself first, and only if that
succeeds attempt the primary (no-buffer) backend, falling back to the
buffered igraph backend if that particular import fails for some other
reason. If `igraph` cannot be imported at all, fall back to the
pure-NetworkX backend. `USING_IGRAPH` reflects which family of backend was
selected (True for either igraph-based backend, False for the NetworkX
backend).

`get_connectivity_column_mappings` (the connectivity output schema) is
always sourced from the NetworkX backend module regardless of which
backend computed the metrics, since all three backends share the same
output column schema and only the NetworkX module defines this function.
"""

try:
    import igraph as ig  # noqa: F401

    try:
        from .connectivity_igraph_primary import calculate_connectivity_metrics
        USING_IGRAPH = True
    except ImportError:
        from .connectivity_igraph_buffered import calculate_connectivity_metrics
        USING_IGRAPH = True
except ImportError:
    from .connectivity_networkx import calculate_connectivity_metrics
    USING_IGRAPH = False

from .connectivity_networkx import get_connectivity_column_mappings

__all__ = [
    "calculate_connectivity_metrics",
    "get_connectivity_column_mappings",
    "USING_IGRAPH",
]
