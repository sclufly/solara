"""
routing
~~~~~~~
Shadow-aware walking route computation for Toronto.

Typical usage::

    from routing import TorontoRoutingEngine

    engine = TorontoRoutingEngine.load()
    report = engine.shadow_route_report(
        start_coord=(-79.3832, 43.6532),
        end_coord=(-79.3800, 43.6600),
        shadow_time=pd.Timestamp("2024-07-15 14:00", tz="America/Toronto"),
    )
"""

from .config import DEFAULT_BUILDINGS_PATH, DEFAULT_NETWORK_NAME, DEFAULT_WALKING_PATHS_DIR, ShadowPenaltyFn
from .engine import TorontoRoutingEngine, compute_shadow_aware_route_report
from .graph import compute_shortest_route, load_graph_bundle, nearest_node
from .plot import plot_route_between_coordinates, plot_shadow_route_report

__all__ = [
    # Engine
    "TorontoRoutingEngine",
    # High-level functions
    "compute_shadow_aware_route_report",
    "compute_shortest_route",
    # Graph utilities
    "load_graph_bundle",
    "nearest_node",
    # Plotting
    "plot_shadow_route_report",
    "plot_route_between_coordinates",
    # Config
    "DEFAULT_NETWORK_NAME",
    "DEFAULT_WALKING_PATHS_DIR",
    "DEFAULT_BUILDINGS_PATH",
    "ShadowPenaltyFn",
]
