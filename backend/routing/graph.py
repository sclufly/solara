"""Graph loading, subgraph construction, and A* path finding."""

import pickle
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point

from .config import (
    DEFAULT_NETWORK_NAME,
    DEFAULT_WALKING_PATHS_DIR,
    ShadowPenaltyFn,
)


# ---------------------------------------------------------------------------
# File path helpers
# ---------------------------------------------------------------------------

def _graph_pickle_path(network_name: str, base_dir: str | Path) -> Path:
    return Path(base_dir) / f"{network_name}-walking-paths.pkl"


def _nodes_parquet_path(network_name: str, base_dir: str | Path) -> Path:
    return Path(base_dir) / f"{network_name}-nodes.parquet"


def _edges_parquet_path(network_name: str, base_dir: str | Path) -> Path:
    return Path(base_dir) / f"{network_name}-edges.parquet"


def _routing_index_path(network_name: str, base_dir: str | Path) -> Path:
    return Path(base_dir) / f"{network_name}-routing-index.pkl"


def _load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------

def load_graph_bundle(
    network_name: str = DEFAULT_NETWORK_NAME,
    base_dir: str | Path = DEFAULT_WALKING_PATHS_DIR,
) -> tuple[nx.MultiDiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, object], dict[str, Any]]:
    """Load the Toronto routing bundle from pickle/Parquet files."""
    paths = {
        "graph":   _graph_pickle_path(network_name, base_dir),
        "nodes":   _nodes_parquet_path(network_name, base_dir),
        "edges":   _edges_parquet_path(network_name, base_dir),
        "index":   _routing_index_path(network_name, base_dir),
    }

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing cached routing files: {', '.join(missing)}")

    graph         = _load_pickle(paths["graph"])
    nodes_gdf     = gpd.read_parquet(paths["nodes"])
    edges_gdf     = gpd.read_parquet(paths["edges"])
    routing_index = _load_pickle(paths["index"])

    if nodes_gdf.empty or edges_gdf.empty:
        raise ValueError("Cached GeoDataFrame layers are empty")
    if nodes_gdf.crs is None or edges_gdf.crs is None:
        raise ValueError("Cached node/edge layers are missing CRS")

    metadata = {
        "network_name":       network_name,
        "crs":                str(edges_gdf.crs),
        "weight_field":       "length_m" if "length_m" in edges_gdf.columns else "length",
        "bundle_format":      "pickle+parquet",
        **{k: str(v) for k, v in paths.items()},
    }
    return graph, nodes_gdf, edges_gdf, metadata, routing_index


# ---------------------------------------------------------------------------
# Node / edge utilities
# ---------------------------------------------------------------------------

def _canonical_node_id(graph: nx.Graph, node_id: object) -> object:
    if node_id in graph:
        return node_id
    if (as_str := str(node_id)) in graph:
        return as_str
    try:
        as_int = int(node_id)
    except (TypeError, ValueError):
        return node_id
    return as_int if as_int in graph else node_id


def nearest_node(
    graph: nx.Graph,
    nodes_gdf: gpd.GeoDataFrame,
    point: Point,
    routing_index: dict[str, Any] | None = None,
) -> object:
    if nodes_gdf.empty:
        raise ValueError("Cannot snap to a node in an empty graph")

    if routing_index is not None:
        _, index = routing_index["node_tree"].query([point.x, point.y], k=1)
        return routing_index["node_ids"][int(index)]

    if "osmid" not in nodes_gdf.columns:
        raise ValueError("nodes layer must contain an 'osmid' column")

    nearest_id = nodes_gdf.loc[nodes_gdf.geometry.distance(point).idxmin(), "osmid"]
    return _canonical_node_id(graph, nearest_id)


def _coerce_point(
    coordinate: tuple[float, float] | list[float] | Point,
    input_crs: str,
    output_crs: str,
) -> Point:
    point = coordinate if isinstance(coordinate, Point) else Point(float(coordinate[0]), float(coordinate[1]))
    if input_crs == output_crs:
        return point
    return gpd.GeoSeries([point], crs=input_crs).to_crs(output_crs).iloc[0]


def _edge_geometry_from_edges_table(
    u: object,
    v: object,
    key: object,
    edge_lookup: dict[tuple[object, object, object], object],
) -> object | None:
    return edge_lookup.get((u, v, key)) or edge_lookup.get((v, u, key))


def _best_edge_choice(
    edge_data: dict[object, dict[str, object]],
    weight_field: str,
    edge_lookup: dict[tuple[object, object, object], object] | None = None,
    u: object | None = None,
    v: object | None = None,
    shadow_penalty_fn: ShadowPenaltyFn | None = None,
) -> tuple[object, dict[str, object], object, float, float]:
    best_key, best_attrs, best_geom, best_cost = None, None, None, float("inf")

    for key, attrs in edge_data.items():
        geom      = None if edge_lookup is None else _edge_geometry_from_edges_table(u, v, key, edge_lookup)
        base_cost = float(attrs.get(weight_field, attrs.get("length", float("inf"))))
        cost      = base_cost + (float(shadow_penalty_fn(geom, attrs)) if shadow_penalty_fn and geom is not None else 0.0)
        if cost < best_cost:
            best_key, best_attrs, best_geom, best_cost = key, attrs, geom, cost

    if best_key is None:
        raise ValueError("No usable edge exists between the requested nodes")

    physical_length = float(best_attrs.get("length", best_attrs.get(weight_field, 0.0)))
    return best_key, best_attrs, best_geom, physical_length, best_cost


# ---------------------------------------------------------------------------
# Subgraph construction
# ---------------------------------------------------------------------------

def _build_corridor(start_point: Point, end_point: Point, buffer_m: float) -> object:
    return LineString([start_point, end_point]).buffer(buffer_m)


def build_route_subgraph(
    graph: nx.Graph,
    edges_gdf: gpd.GeoDataFrame,
    routing_index: dict[str, Any],
    start_point: Point,
    end_point: Point,
    buffer_m: float,
) -> tuple[nx.MultiDiGraph, dict[tuple[object, object, object], object]]:
    """Return a subgraph and edge-geometry lookup for a straight-line corridor."""
    corridor      = _build_corridor(start_point, end_point, buffer_m)
    candidate_idx = list(routing_index["edge_sindex"].query(corridor))

    if not candidate_idx:
        return graph.copy(), {}

    route_edges: list[tuple] = []
    edge_lookup: dict        = {}

    for row in edges_gdf.iloc[candidate_idx].itertuples():
        u   = _canonical_node_id(graph, row.u)
        v   = _canonical_node_id(graph, row.v)
        key = row.key
        route_edges.append((u, v, key))
        edge_lookup[(u, v, key)] = row.geometry
        edge_lookup[(v, u, key)] = row.geometry

    if not route_edges:
        return graph.copy(), {}

    return graph.edge_subgraph(route_edges).copy(), edge_lookup


def build_narrow_corridor_subgraph(
    graph: nx.Graph,
    edges_gdf: gpd.GeoDataFrame,
    routing_index: dict,
    fastest_route: dict,
    narrow_buffer_m: float = 100.0,
) -> tuple[nx.MultiDiGraph, dict]:
    """Subgraph restricted to a narrow corridor around an existing route."""
    route_geometries = [e["geometry"] for e in fastest_route["route_edges"] if e.get("geometry") is not None]
    if not route_geometries:
        return graph.copy(), {}

    corridor      = gpd.GeoSeries(route_geometries, crs=edges_gdf.crs).union_all().buffer(narrow_buffer_m)
    candidate_idx = list(routing_index["edge_sindex"].query(corridor))

    route_edges: list[tuple] = []
    edge_lookup: dict        = {}

    for row in edges_gdf.iloc[candidate_idx].itertuples():
        u   = _canonical_node_id(graph, row.u)
        v   = _canonical_node_id(graph, row.v)
        key = row.key
        route_edges.append((u, v, key))
        edge_lookup[(u, v, key)] = row.geometry
        edge_lookup[(v, u, key)] = row.geometry

    if not route_edges:
        return graph.copy(), {}

    return graph.edge_subgraph(route_edges).copy(), edge_lookup


# ---------------------------------------------------------------------------
# A* helpers
# ---------------------------------------------------------------------------

def _astar_heuristic(graph: nx.Graph, target_node: object) -> Callable:
    tx, ty = graph.nodes[target_node]["x"], graph.nodes[target_node]["y"]

    def heuristic(node: object, _: object) -> float:
        return float(np.hypot(graph.nodes[node]["x"] - tx, graph.nodes[node]["y"] - ty))

    return heuristic


def _make_astar_weight(
    weight_field: str,
    edge_lookup: dict,
    shadow_penalty_fn: ShadowPenaltyFn | None = None,
) -> Callable:
    def weight(u: object, v: object, edge_data: dict) -> float:
        _, _, _, _, best_cost = _best_edge_choice(
            edge_data=edge_data,
            weight_field=weight_field,
            edge_lookup=edge_lookup,
            u=u,
            v=v,
            shadow_penalty_fn=shadow_penalty_fn,
        )
        return best_cost

    return weight


# ---------------------------------------------------------------------------
# Public routing function
# ---------------------------------------------------------------------------

def compute_shortest_route(
    graph: nx.Graph,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    start_coord: tuple[float, float] | list[float] | Point,
    end_coord: tuple[float, float] | list[float] | Point,
    input_crs: str = "EPSG:4326",
    weight_field: str = "length",
    routing_index: dict[str, Any] | None = None,
    route_buffer_m: float = 250.0,
    shadow_penalty_fn: ShadowPenaltyFn | None = None,
    corridor_retry_multiplier: float = 2.0,
    corridor_retries: int = 2,
) -> dict[str, object]:
    """Compute a shortest (or shadow-aware) route between two coordinates."""
    network_crs = str(edges_gdf.crs)
    start_point = _coerce_point(start_coord, input_crs, network_crs)
    end_point   = _coerce_point(end_coord,   input_crs, network_crs)
    start_node  = nearest_node(graph, nodes_gdf, start_point, routing_index)
    end_node    = nearest_node(graph, nodes_gdf, end_point,   routing_index)

    route_graph  = graph
    edge_lookup  = {}
    route_nodes  = None

    if routing_index is not None:
        current_buffer_m = route_buffer_m
        for _ in range(max(1, corridor_retries + 1)):
            route_graph, edge_lookup = build_route_subgraph(
                graph, edges_gdf, routing_index, start_point, end_point, current_buffer_m,
            )
            try:
                route_nodes = nx.astar_path(
                    route_graph, start_node, end_node,
                    heuristic=_astar_heuristic(route_graph, end_node),
                    weight=_make_astar_weight(weight_field, edge_lookup, shadow_penalty_fn),
                )
                break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                current_buffer_m *= corridor_retry_multiplier

    if route_nodes is None:
        route_nodes = nx.astar_path(
            route_graph, start_node, end_node,
            heuristic=_astar_heuristic(route_graph, end_node),
            weight=weight_field,
        )

    route_edges: list[dict] = []
    route_length_m = 0.0
    route_cost_m   = 0.0

    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        edge_data = route_graph.get_edge_data(u, v)
        if not edge_data:
            raise ValueError(f"No edge exists between nodes {u} and {v}")

        key, attrs, edge_geom, segment_length, segment_cost = _best_edge_choice(
            edge_data, weight_field, edge_lookup, u, v, shadow_penalty_fn,
        )
        route_length_m += segment_length
        route_cost_m   += segment_cost
        route_edges.append({"u": u, "v": v, "key": key, "length_m": segment_length, "cost_m": segment_cost, "geometry": edge_geom})

    return {
        "start_node":    start_node,
        "end_node":      end_node,
        "start_point":   start_point,
        "end_point":     end_point,
        "route_nodes":   route_nodes,
        "route_edges":   route_edges,
        "route_length_m": route_length_m,
        "route_cost_m":  route_cost_m,
    }
