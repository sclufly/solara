from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
from scipy.spatial import cKDTree
from shapely.geometry import Point


DEFAULT_NETWORK_NAME = "toronto"
DEFAULT_WALKING_PATHS_DIR = Path("data/walking-paths")


def _graphml_path(network_name: str, base_dir: str | Path = DEFAULT_WALKING_PATHS_DIR) -> Path:
    return Path(base_dir) / f"{network_name}-walking-paths.graphml"


def _gpkg_path(network_name: str, base_dir: str | Path = DEFAULT_WALKING_PATHS_DIR) -> Path:
    return Path(base_dir) / f"{network_name}-network.gpkg"


def _canonical_node_id(graph: nx.Graph, node_id: object) -> object:
    if node_id in graph:
        return node_id

    as_str = str(node_id)
    if as_str in graph:
        return as_str

    try:
        as_int = int(node_id)
    except (TypeError, ValueError):
        as_int = None

    if as_int is not None and as_int in graph:
        return as_int

    return node_id


def build_routing_index(
    graph: nx.Graph,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
) -> dict[str, Any]:
    """Build one-time indexes to accelerate snapping and edge-geometry retrieval."""

    node_id_column = "osmid" if "osmid" in nodes_gdf.columns else None
    if node_id_column is None:
        raise ValueError("nodes layer must contain an 'osmid' column")

    if "key" not in edges_gdf.columns:
        raise ValueError("edges layer must contain a 'key' column")

    node_ids_raw = nodes_gdf[node_id_column].to_list()
    canonical_node_ids = [_canonical_node_id(graph, raw_id) for raw_id in node_ids_raw]

    x_values = nodes_gdf.geometry.x.to_numpy(dtype=float)
    y_values = nodes_gdf.geometry.y.to_numpy(dtype=float)
    points_array = np.column_stack((x_values, y_values))

    if len(points_array) == 0:
        raise ValueError("Cannot build routing index from an empty nodes layer")

    node_tree = cKDTree(points_array)

    edge_geom_by_uvk: dict[tuple[str, str, str], object] = {}
    u_values = edges_gdf["u"].astype(str).to_numpy()
    v_values = edges_gdf["v"].astype(str).to_numpy()
    k_values = edges_gdf["key"].astype(str).to_numpy()
    geom_values = edges_gdf.geometry.to_numpy()

    for u, v, k, geom in zip(u_values, v_values, k_values, geom_values):
        edge_geom_by_uvk[(u, v, k)] = geom
        edge_geom_by_uvk[(v, u, k)] = geom

    return {
        "node_tree": node_tree,
        "node_ids": canonical_node_ids,
        "edge_geom_by_uvk": edge_geom_by_uvk,
    }


def load_graph_bundle(
    network_name: str = DEFAULT_NETWORK_NAME,
    base_dir: str | Path = DEFAULT_WALKING_PATHS_DIR,
) -> tuple[nx.MultiDiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, object]]:
    """Load graph topology from GraphML and geometries from GeoPackage."""

    graphml_path = _graphml_path(network_name, base_dir=base_dir)
    gpkg_path = _gpkg_path(network_name, base_dir=base_dir)

    if not graphml_path.exists():
        raise FileNotFoundError(f"GraphML file not found: {graphml_path}")
    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage file not found: {gpkg_path}")

    graph = ox.load_graphml(graphml_path)
    nodes_gdf = gpd.read_file(gpkg_path, layer="nodes")
    edges_gdf = gpd.read_file(gpkg_path, layer="edges")

    if nodes_gdf.empty or edges_gdf.empty:
        raise ValueError("GeoPackage layers are empty")
    if nodes_gdf.crs is None or edges_gdf.crs is None:
        raise ValueError("GeoPackage node/edge layers are missing CRS")

    metadata = {
        "network_name": network_name,
        "graphml_path": str(graphml_path),
        "gpkg_path": str(gpkg_path),
        "crs": str(edges_gdf.crs),
        "weight_field": "length_m" if "length_m" in edges_gdf.columns else "length",
    }
    return graph, nodes_gdf, edges_gdf, metadata


def nearest_node(graph: nx.Graph, nodes_gdf: gpd.GeoDataFrame, point: Point, routing_index: dict[str, Any] | None = None) -> object:
    if nodes_gdf.empty:
        raise ValueError("Cannot snap to a node in an empty graph")

    if routing_index is not None:
        _distance, index = routing_index["node_tree"].query([point.x, point.y], k=1)
        return routing_index["node_ids"][int(index)]

    node_id_column = "osmid" if "osmid" in nodes_gdf.columns else None
    if node_id_column is None:
        raise ValueError("nodes layer must contain an 'osmid' column")

    distances = nodes_gdf.geometry.distance(point)
    nearest_id = nodes_gdf.loc[distances.idxmin(), node_id_column]
    return _canonical_node_id(graph, nearest_id)


def _coerce_point(
    coordinate: tuple[float, float] | list[float] | Point,
    input_crs: str,
    output_crs: str,
) -> Point:
    if isinstance(coordinate, Point):
        point = coordinate
    else:
        point = Point(float(coordinate[0]), float(coordinate[1]))

    if input_crs == output_crs:
        return point

    point_series = gpd.GeoSeries([point], crs=input_crs)
    return point_series.to_crs(output_crs).iloc[0]


def _edge_geometry_from_edges_table(
    edges_gdf: gpd.GeoDataFrame,
    u: object,
    v: object,
    key: object,
    routing_index: dict[str, Any] | None = None,
):
    if routing_index is not None:
        return routing_index["edge_geom_by_uvk"].get((str(u), str(v), str(key)))

    key_col = "key" if "key" in edges_gdf.columns else None
    if key_col is None:
        return None

    key_str = str(key)
    u_str = str(u)
    v_str = str(v)

    direct = edges_gdf[
        (edges_gdf["u"].astype(str) == u_str)
        & (edges_gdf["v"].astype(str) == v_str)
        & (edges_gdf[key_col].astype(str) == key_str)
    ]
    if not direct.empty:
        return direct.geometry.iloc[0]

    reverse = edges_gdf[
        (edges_gdf["u"].astype(str) == v_str)
        & (edges_gdf["v"].astype(str) == u_str)
        & (edges_gdf[key_col].astype(str) == key_str)
    ]
    if not reverse.empty:
        return reverse.geometry.iloc[0]

    return None


def _edge_for_node_pair(
    graph: nx.Graph,
    edges_gdf: gpd.GeoDataFrame,
    u: object,
    v: object,
    weight_field: str,
    routing_index: dict[str, Any] | None = None,
) -> tuple[object, dict[str, object], object]:
    edge_data = graph.get_edge_data(u, v)
    if not edge_data:
        raise ValueError(f"No edge exists between nodes {u} and {v}")

    key, attrs = min(
        edge_data.items(),
        key=lambda item: float(item[1].get(weight_field, item[1].get("length", float("inf")))),
    )
    geom = _edge_geometry_from_edges_table(edges_gdf, u=u, v=v, key=key, routing_index=routing_index)
    return key, attrs, geom


def compute_shortest_route(
    graph: nx.Graph,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    start_coord: tuple[float, float] | list[float] | Point,
    end_coord: tuple[float, float] | list[float] | Point,
    input_crs: str = "EPSG:4326",
    weight_field: str = "length",
    routing_index: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Compute an edge-length shortest path route between two coordinates."""

    network_crs = str(edges_gdf.crs)
    start_point = _coerce_point(start_coord, input_crs=input_crs, output_crs=network_crs)
    end_point = _coerce_point(end_coord, input_crs=input_crs, output_crs=network_crs)

    start_node = nearest_node(graph, nodes_gdf, start_point, routing_index=routing_index)
    end_node = nearest_node(graph, nodes_gdf, end_point, routing_index=routing_index)

    route_nodes = nx.shortest_path(graph, start_node, end_node, weight=weight_field)
    route_edges: list[dict[str, object]] = []
    route_length_m = 0.0

    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        key, attrs, edge_geom = _edge_for_node_pair(
            graph,
            edges_gdf=edges_gdf,
            u=u,
            v=v,
            weight_field=weight_field,
            routing_index=routing_index,
        )
        segment_length = float(attrs.get(weight_field, attrs.get("length", 0.0)))
        route_length_m += segment_length
        route_edges.append(
            {
                "u": u,
                "v": v,
                "key": key,
                "length_m": segment_length,
                "geometry": edge_geom,
            }
        )

    return {
        "start_node": start_node,
        "end_node": end_node,
        "start_point": start_point,
        "end_point": end_point,
        "route_nodes": route_nodes,
        "route_edges": route_edges,
        "route_length_m": route_length_m,
    }


def plot_route_between_coordinates(
    graph: nx.Graph,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    start_coord: tuple[float, float] | list[float] | Point,
    end_coord: tuple[float, float] | list[float] | Point,
    input_crs: str = "EPSG:4326",
    weight_field: str = "length",
    routing_index: dict[str, Any] | None = None,
    full_network: bool = False,
    route_buffer_m: float = 150.0,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, dict[str, object]]:
    route = compute_shortest_route(
        graph=graph,
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        start_coord=start_coord,
        end_coord=end_coord,
        input_crs=input_crs,
        weight_field=weight_field,
        routing_index=routing_index,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig = ax.figure

    route_geometries = [edge["geometry"] for edge in route["route_edges"] if edge["geometry"] is not None]
    if route_geometries:
        route_series = gpd.GeoSeries(route_geometries, crs=edges_gdf.crs)
    else:
        route_series = gpd.GeoSeries([], crs=edges_gdf.crs)

    if full_network or route_series.empty:
        edges_to_plot = edges_gdf
    else:
        route_area = route_series.unary_union.buffer(route_buffer_m)
        edges_to_plot = edges_gdf[edges_gdf.geometry.intersects(route_area)]

    edges_to_plot.plot(ax=ax, color="#bcbcbc", linewidth=0.4, alpha=0.5, zorder=1)

    if not route_series.empty:
        route_series.plot(
            ax=ax,
            color="#d7301f",
            linewidth=2.5,
            alpha=0.95,
            zorder=3,
        )

    start_geom = route["start_point"]
    end_geom = route["end_point"]
    ax.scatter([start_geom.x], [start_geom.y], s=55, c="#1a9850", edgecolors="black", linewidths=0.6, zorder=4, label="start")
    ax.scatter([end_geom.x], [end_geom.y], s=55, c="#2b83ba", edgecolors="black", linewidths=0.6, zorder=4, label="end")

    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.legend(loc="upper right")
    ax.set_title(f"Shortest route by length: {route['route_length_m']:.1f} m")
    return fig, ax, route


def summarize_graph(graph: nx.Graph) -> str:
    return f"nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}, crs={graph.graph.get('crs')}"


if __name__ == "__main__":
    graph, nodes_gdf, edges_gdf, metadata = load_graph_bundle(network_name=DEFAULT_NETWORK_NAME)
    routing_index = build_routing_index(graph=graph, nodes_gdf=nodes_gdf, edges_gdf=edges_gdf)
    print(summarize_graph(graph))
    print(f"nodes table rows={len(nodes_gdf)}")
    print(f"edges table rows={len(edges_gdf)}")
    print(f"dataset={metadata['network_name']}")
    print(f"weight field={metadata['weight_field']}")

    start_coord = (-79.385882, 43.642017)
    end_coord = (-79.388826, 43.644417)

    fig, ax, route = plot_route_between_coordinates(
        graph=graph,
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        start_coord=start_coord,
        end_coord=end_coord,
        input_crs="EPSG:4326",
        weight_field=str(metadata["weight_field"]),
        routing_index=routing_index,
        full_network=False,
        route_buffer_m=150.0,
    )
    print(f"route nodes={len(route['route_nodes'])}")
    print(f"route length m={route['route_length_m']:.1f}")
    plt.show()