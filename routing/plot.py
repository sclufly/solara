"""Matplotlib visualisation for routes and shadow reports."""

from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from shapely.geometry import Point

from .config import ShadowPenaltyFn
from .graph import compute_shortest_route


def plot_shadow_route_report(
    report: dict[str, object],
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot buildings, shadows, and both routes from a shadow route report."""
    buildings_gdf = report["buildings_gdf"]
    shadows_gdf   = report["shadows_gdf"]
    fastest_route = report["fastest_route"]
    shaded_route  = report["shaded_route"]

    fig, ax = _get_or_create_axes(ax)

    if isinstance(buildings_gdf, gpd.GeoDataFrame) and not buildings_gdf.empty:
        buildings_gdf.plot(ax=ax, color="#c7c7c7", edgecolor="none", alpha=0.35, zorder=1)
    if isinstance(shadows_gdf, gpd.GeoDataFrame) and not shadows_gdf.empty:
        shadows_gdf.plot(ax=ax, color="#1d4ed8", edgecolor="none", alpha=0.22, zorder=2)

    ax.add_collection(LineCollection(_route_segments(fastest_route), linewidths=2.0, colors="#111827", alpha=0.8,  zorder=4))
    ax.add_collection(LineCollection(_route_segments(shaded_route),  linewidths=3.0, colors="#dc2626", alpha=0.95, zorder=5))

    _plot_endpoints(ax, fastest_route)

    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.legend(loc="upper right")
    ax.set_title(
        f"Fastest: {fastest_route['route_length_m']:.1f} m  |  "
        f"Shade-aware: {shaded_route['route_length_m']:.1f} m"
    )
    return fig, ax


def plot_route_between_coordinates(
    graph: nx.Graph,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    start_coord: tuple[float, float] | list[float] | Point,
    end_coord: tuple[float, float] | list[float] | Point,
    input_crs: str = "EPSG:4326",
    weight_field: str = "length",
    routing_index: dict[str, Any] | None = None,
    shadow_penalty_fn: ShadowPenaltyFn | None = None,
    route_buffer_m: float = 150.0,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, dict[str, object]]:
    """Compute and plot a single route, with the local street network as context."""
    route = compute_shortest_route(
        graph=graph,
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        start_coord=start_coord,
        end_coord=end_coord,
        input_crs=input_crs,
        weight_field=weight_field,
        routing_index=routing_index,
        route_buffer_m=route_buffer_m,
        shadow_penalty_fn=shadow_penalty_fn,
    )

    fig, ax        = _get_or_create_axes(ax)
    route_geoms    = [e["geometry"] for e in route["route_edges"] if e["geometry"] is not None]
    edges_to_plot  = _local_edges(edges_gdf, routing_index, route_geoms, route_buffer_m)

    ax.add_collection(LineCollection(_coords(edges_to_plot.geometry), linewidths=0.4, alpha=0.5))
    ax.add_collection(LineCollection(_coords(route_geoms),            linewidths=2.5))

    _plot_endpoints(ax, route)

    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.legend(loc="upper right")
    ax.set_title(f"Shortest route: {route.get('route_cost_m', route['route_length_m']):.1f} m")
    return fig, ax, route


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_or_create_axes(ax: plt.Axes | None) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        return plt.subplots(figsize=(12, 12))
    return ax.figure, ax


def _route_segments(route: dict) -> list[np.ndarray]:
    return [
        np.asarray(e["geometry"].coords)
        for e in route["route_edges"]
        if e.get("geometry") is not None and hasattr(e["geometry"], "coords")
    ]


def _coords(geometries) -> list[np.ndarray]:
    return [np.asarray(g.coords) for g in geometries if g is not None and hasattr(g, "coords")]


def _plot_endpoints(ax: plt.Axes, route: dict) -> None:
    ax.scatter([route["start_point"].x], [route["start_point"].y], s=55, zorder=6, label="start")
    ax.scatter([route["end_point"].x],   [route["end_point"].y],   s=55, zorder=6, label="end")


def _local_edges(
    edges_gdf: gpd.GeoDataFrame,
    routing_index: dict | None,
    route_geoms: list,
    buffer_m: float,
) -> gpd.GeoDataFrame:
    if not route_geoms or routing_index is None:
        return edges_gdf
    route_area    = gpd.GeoSeries(route_geoms, crs=edges_gdf.crs).union_all().buffer(buffer_m)
    candidate_idx = list(routing_index["edge_sindex"].query(route_area))
    candidates    = edges_gdf.iloc[candidate_idx]
    return candidates[candidates.geometry.intersects(route_area)]
