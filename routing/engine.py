"""High-level routing orchestration: shadow-aware route reports and the long-lived engine."""

from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point

from .config import DEFAULT_BUILDINGS_PATH, DEFAULT_NETWORK_NAME, DEFAULT_WALKING_PATHS_DIR
from .graph import (
    build_narrow_corridor_subgraph,
    compute_shortest_route,
    load_graph_bundle,
    _coerce_point,
)
from .shadows import (
    annotate_edges_with_shadow,
    build_shadow_union,
    load_buildings_and_shadows,
    rasterize_shadow_union,
    route_shadow_fraction,
)


def compute_shadow_aware_route_report(
    graph: nx.Graph,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    start_coord: tuple[float, float] | list[float] | Point,
    end_coord: tuple[float, float] | list[float] | Point,
    shadow_time: pd.Timestamp,
    buildings_path: str | Path = DEFAULT_BUILDINGS_PATH,
    input_crs: str = "EPSG:4326",
    route_buffer_m: float = 250.0,
    shadow_read_buffer_m: float = 200.0,
    shade_weight: float = 0.75,
    narrow_buffer_m: float = 100.0,
    routing_index: dict[str, Any] | None = None,
) -> dict[str, object]:
    """
    Compute both the fastest and most-shaded walking routes between two points.

    Returns a report dict containing both routes, shadow geometries, and shade
    coverage percentages for each route.
    """
    network_crs = str(edges_gdf.crs)
    start_point = _coerce_point(start_coord, input_crs, network_crs)
    end_point   = _coerce_point(end_coord,   input_crs, network_crs)
    corridor    = LineString([start_point, end_point]).buffer(route_buffer_m)

    buildings_gdf, shadows_gdf = load_buildings_and_shadows(
        buildings_path=buildings_path,
        corridor=corridor,
        route_crs=network_crs,
        shadow_time=shadow_time,
        read_buffer_m=shadow_read_buffer_m,
    )

    shadow_union = build_shadow_union(shadows_gdf)

    fastest_route = compute_shortest_route(
        graph=graph,
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        start_coord=start_coord,
        end_coord=end_coord,
        input_crs=input_crs,
        weight_field="length",
        routing_index=routing_index,
        route_buffer_m=route_buffer_m,
    )

    if shadow_union is not None:
        grid, transform = rasterize_shadow_union(shadow_union)

        route_graph, edge_lookup = build_narrow_corridor_subgraph(
            graph=graph,
            edges_gdf=edges_gdf,
            routing_index=routing_index,
            fastest_route=fastest_route,
            narrow_buffer_m=narrow_buffer_m,
        )
        annotate_edges_with_shadow(route_graph, edge_lookup, grid, transform, "length", shade_weight)

        shaded_route = compute_shortest_route(
            graph=route_graph,
            nodes_gdf=nodes_gdf,
            edges_gdf=edges_gdf,
            start_coord=start_coord,
            end_coord=end_coord,
            input_crs=input_crs,
            weight_field="shaded_cost",
            routing_index=routing_index,
            route_buffer_m=route_buffer_m,
            shadow_penalty_fn=None,
        )
    else:
        shaded_route = fastest_route

    return {
        "shadow_time":       shadow_time,
        "buildings_gdf":     buildings_gdf,
        "shadows_gdf":       shadows_gdf,
        "shadow_union":      shadow_union,
        "fastest_route":     fastest_route,
        "shaded_route":      shaded_route,
        "fastest_shade_pct": route_shadow_fraction(fastest_route, shadow_union),
        "shaded_shade_pct":  route_shadow_fraction(shaded_route,  shadow_union),
    }


class TorontoRoutingEngine:
    """Long-lived router that keeps the graph bundle hot in memory."""

    def __init__(
        self,
        network_name: str = DEFAULT_NETWORK_NAME,
        base_dir: str | Path = DEFAULT_WALKING_PATHS_DIR,
    ) -> None:
        self.graph, self.nodes_gdf, self.edges_gdf, self.metadata, self.routing_index = (
            load_graph_bundle(network_name=network_name, base_dir=base_dir)
        )

    @classmethod
    def load(
        cls,
        network_name: str = DEFAULT_NETWORK_NAME,
        base_dir: str | Path = DEFAULT_WALKING_PATHS_DIR,
    ) -> "TorontoRoutingEngine":
        return cls(network_name=network_name, base_dir=base_dir)

    def route(self, *args, **kwargs) -> dict[str, object]:
        """Compute a shortest route. Accepts all compute_shortest_route kwargs."""
        return compute_shortest_route(
            graph=self.graph,
            nodes_gdf=self.nodes_gdf,
            edges_gdf=self.edges_gdf,
            routing_index=self.routing_index,
            *args,
            **kwargs,
        )

    def shadow_route_report(self, *args, **kwargs) -> dict[str, object]:
        """Compute a shadow-aware route report. Accepts all compute_shadow_aware_route_report kwargs."""
        return compute_shadow_aware_route_report(
            graph=self.graph,
            nodes_gdf=self.nodes_gdf,
            edges_gdf=self.edges_gdf,
            routing_index=self.routing_index,
            *args,
            **kwargs,
        )
