import pickle
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point


DEFAULT_NETWORK_NAME = "toronto"
DEFAULT_WALKING_PATHS_DIR = Path("data/walking-paths")
ShadowPenaltyFn = Callable[[object, dict[str, object]], float]


def _graph_pickle_path(network_name: str, base_dir: str | Path = DEFAULT_WALKING_PATHS_DIR) -> Path:
    return Path(base_dir) / f"{network_name}-walking-paths.pkl"


def _nodes_parquet_path(network_name: str, base_dir: str | Path = DEFAULT_WALKING_PATHS_DIR) -> Path:
    return Path(base_dir) / f"{network_name}-nodes.parquet"


def _edges_parquet_path(network_name: str, base_dir: str | Path = DEFAULT_WALKING_PATHS_DIR) -> Path:
    return Path(base_dir) / f"{network_name}-edges.parquet"


def _routing_index_path(network_name: str, base_dir: str | Path = DEFAULT_WALKING_PATHS_DIR) -> Path:
    return Path(base_dir) / f"{network_name}-routing-index.pkl"


def _load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


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


def load_graph_bundle(
    network_name: str = DEFAULT_NETWORK_NAME,
    base_dir: str | Path = DEFAULT_WALKING_PATHS_DIR,
) -> tuple[nx.MultiDiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, object], dict[str, Any]]:
    """Load the Toronto routing bundle from pickle/Parquet files."""

    graph_pickle_path = _graph_pickle_path(network_name, base_dir=base_dir)
    nodes_parquet_path = _nodes_parquet_path(network_name, base_dir=base_dir)
    edges_parquet_path = _edges_parquet_path(network_name, base_dir=base_dir)
    routing_index_path = _routing_index_path(network_name, base_dir=base_dir)

    missing = [path for path in [graph_pickle_path, nodes_parquet_path, edges_parquet_path, routing_index_path] if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing cached routing files: {missing_list}")

    graph = _load_pickle(graph_pickle_path)
    nodes_gdf = gpd.read_parquet(nodes_parquet_path)
    edges_gdf = gpd.read_parquet(edges_parquet_path)
    routing_index = _load_pickle(routing_index_path)

    if nodes_gdf.empty or edges_gdf.empty:
        raise ValueError("Cached GeoDataFrame layers are empty")
    if nodes_gdf.crs is None or edges_gdf.crs is None:
        raise ValueError("Cached node/edge layers are missing CRS")

    metadata = {
        "network_name": network_name,
        "graph_pickle_path": str(graph_pickle_path),
        "nodes_parquet_path": str(nodes_parquet_path),
        "edges_parquet_path": str(edges_parquet_path),
        "routing_index_path": str(routing_index_path),
        "crs": str(edges_gdf.crs),
        "weight_field": "length_m" if "length_m" in edges_gdf.columns else "length",
        "bundle_format": "pickle+parquet",
    }
    return graph, nodes_gdf, edges_gdf, metadata, routing_index


def _edge_geometry_from_edges_table(
    u: object,
    v: object,
    key: object,
    edge_lookup: dict[tuple[object, object, object], object],
):
    geom = edge_lookup.get((u, v, key))
    if geom is not None:
        return geom
    return edge_lookup.get((v, u, key))


def _best_edge_choice(
    edge_data: dict[object, dict[str, object]],
    weight_field: str,
    edge_lookup: dict[tuple[object, object, object], object] | None = None,
    u: object | None = None,
    v: object | None = None,
    shadow_penalty_fn: ShadowPenaltyFn | None = None,
) -> tuple[object, dict[str, object], object, float, float]:
    best_key: object | None = None
    best_attrs: dict[str, object] | None = None
    best_geom: object | None = None
    best_cost = float("inf")

    for key, attrs in edge_data.items():
        geom = None if edge_lookup is None else _edge_geometry_from_edges_table(u, v, key, edge_lookup)
        base_cost = float(attrs.get(weight_field, attrs.get("length", float("inf"))))
        cost = base_cost + (float(shadow_penalty_fn(geom, attrs)) if shadow_penalty_fn and geom is not None else 0.0)
        if cost < best_cost:
            best_key = key
            best_attrs = attrs
            best_geom = geom
            best_cost = cost

    if best_key is None or best_attrs is None:
        raise ValueError("No usable edge exists between the requested nodes")

    return best_key, best_attrs, best_geom, float(best_attrs.get(weight_field, best_attrs.get("length", 0.0))), best_cost


def _build_route_corridor(
    start_point: Point,
    end_point: Point,
    buffer_m: float,
) -> object:
    return LineString([start_point, end_point]).buffer(buffer_m)


def _route_search_graph(
    graph: nx.Graph,
    edges_gdf: gpd.GeoDataFrame,
    routing_index: dict[str, Any],
    start_point: Point,
    end_point: Point,
    buffer_m: float,
) -> tuple[nx.MultiDiGraph, dict[tuple[object, object, object], object]]:
    corridor = _build_route_corridor(start_point, end_point, buffer_m)
    candidate_idx = list(routing_index["edge_sindex"].query(corridor))

    if not candidate_idx:
        return graph.copy(), {}

    candidate_edges = edges_gdf.iloc[candidate_idx]
    route_edges: list[tuple[object, object, object]] = []
    edge_lookup: dict[tuple[object, object, object], object] = {}

    for row in candidate_edges.itertuples():
        u = _canonical_node_id(graph, row.u)
        v = _canonical_node_id(graph, row.v)
        key = row.key
        route_edges.append((u, v, key))
        edge_lookup[(u, v, key)] = row.geometry
        edge_lookup[(v, u, key)] = row.geometry

    if not route_edges:
        return graph.copy(), {}

    return graph.edge_subgraph(route_edges).copy(), edge_lookup


def _make_astar_weight(
    weight_field: str,
    edge_lookup: dict[tuple[object, object, object], object],
    shadow_penalty_fn: ShadowPenaltyFn | None = None,
) -> Callable[[object, object, dict[object, dict[str, object]]], float]:
    def weight(u: object, v: object, edge_data: dict[object, dict[str, object]]) -> float:
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


def nearest_node(graph: nx.Graph, nodes_gdf: gpd.GeoDataFrame, point: Point, routing_index: dict[str, Any] | None = None) -> object:
    if nodes_gdf.empty:
        raise ValueError("Cannot snap to a node in an empty graph")

    if routing_index is not None:
        _distance, index = routing_index["node_tree"].query([point.x, point.y], k=1)
        return routing_index["node_ids"][int(index)]

    if "osmid" not in nodes_gdf.columns:
        raise ValueError("nodes layer must contain an 'osmid' column")

    distances = nodes_gdf.geometry.distance(point)
    nearest_id = nodes_gdf.loc[distances.idxmin(), "osmid"]
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

    return gpd.GeoSeries([point], crs=input_crs).to_crs(output_crs).iloc[0]


def _astar_heuristic(graph: nx.Graph, target_node: object):
    tx = graph.nodes[target_node]["x"]
    ty = graph.nodes[target_node]["y"]

    def heuristic(node: object, _: object) -> float:
        nx_ = graph.nodes[node]["x"]
        ny_ = graph.nodes[node]["y"]
        return float(np.hypot(nx_ - tx, ny_ - ty))

    return heuristic


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
    """Compute a shortest route between two coordinates."""

    network_crs = str(edges_gdf.crs)
    start_point = _coerce_point(start_coord, input_crs=input_crs, output_crs=network_crs)
    end_point = _coerce_point(end_coord, input_crs=input_crs, output_crs=network_crs)

    start_node = nearest_node(graph, nodes_gdf, start_point, routing_index=routing_index)
    end_node = nearest_node(graph, nodes_gdf, end_point, routing_index=routing_index)

    route_graph = graph
    edge_lookup: dict[tuple[object, object, object], object] = {}
    route_nodes: list[object] | None = None

    if routing_index is not None:
        current_buffer_m = route_buffer_m
        for _ in range(max(1, corridor_retries + 1)):
            route_graph, edge_lookup = _route_search_graph(
                graph=graph,
                edges_gdf=edges_gdf,
                routing_index=routing_index,
                start_point=start_point,
                end_point=end_point,
                buffer_m=current_buffer_m,
            )
            try:
                route_nodes = nx.astar_path(
                    route_graph,
                    start_node,
                    end_node,
                    heuristic=_astar_heuristic(route_graph, end_node),
                    weight=_make_astar_weight(weight_field=weight_field, edge_lookup=edge_lookup, shadow_penalty_fn=shadow_penalty_fn),
                )
                break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                current_buffer_m *= corridor_retry_multiplier

    if route_nodes is None:
        route_nodes = nx.astar_path(route_graph, start_node, end_node, heuristic=_astar_heuristic(route_graph, end_node), weight=weight_field)

    route_edges: list[dict[str, object]] = []
    route_length_m = 0.0
    route_cost_m = 0.0

    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        edge_data = route_graph.get_edge_data(u, v)
        if not edge_data:
            raise ValueError(f"No edge exists between nodes {u} and {v}")

        key, attrs, edge_geom, segment_length, segment_cost = _best_edge_choice(
            edge_data=edge_data,
            weight_field=weight_field,
            edge_lookup=edge_lookup,
            u=u,
            v=v,
            shadow_penalty_fn=shadow_penalty_fn,
        )
        route_length_m += segment_length
        route_cost_m += segment_cost
        route_edges.append(
            {
                "u": u,
                "v": v,
                "key": key,
                "length_m": segment_length,
                "cost_m": segment_cost,
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
        "route_cost_m": route_cost_m,
    }


class TorontoRoutingEngine:
    """Long-lived Toronto-only router that keeps the bundle hot in memory."""

    def __init__(self, network_name: str = DEFAULT_NETWORK_NAME, base_dir: str | Path = DEFAULT_WALKING_PATHS_DIR):
        self.graph, self.nodes_gdf, self.edges_gdf, self.metadata, self.routing_index = load_graph_bundle(
            network_name=network_name,
            base_dir=base_dir,
        )

    @classmethod
    def load(cls, network_name: str = DEFAULT_NETWORK_NAME, base_dir: str | Path = DEFAULT_WALKING_PATHS_DIR) -> "TorontoRoutingEngine":
        return cls(network_name=network_name, base_dir=base_dir)

    def route(self, *args, **kwargs) -> dict[str, object]:
        return compute_shortest_route(
            graph=self.graph,
            nodes_gdf=self.nodes_gdf,
            edges_gdf=self.edges_gdf,
            routing_index=self.routing_index,
            *args,
            **kwargs,
        )


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
        route_buffer_m=route_buffer_m,
        shadow_penalty_fn=shadow_penalty_fn,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig = ax.figure

    route_geometries = [edge["geometry"] for edge in route["route_edges"] if edge["geometry"] is not None]

    if full_network or not route_geometries:
        edges_to_plot = edges_gdf
    else:
        route_area = gpd.GeoSeries(route_geometries, crs=edges_gdf.crs).union_all().buffer(route_buffer_m)
        candidate_idx = list(routing_index["edge_sindex"].query(route_area))
        edges_to_plot = edges_gdf.iloc[candidate_idx]
        edges_to_plot = edges_to_plot[edges_to_plot.geometry.intersects(route_area)]

    ax.add_collection(LineCollection([np.asarray(geom.coords) for geom in edges_to_plot.geometry if geom is not None and hasattr(geom, "coords")], linewidths=0.4, alpha=0.5))
    ax.add_collection(LineCollection([np.asarray(geom.coords) for geom in route_geometries if geom is not None and hasattr(geom, "coords")], linewidths=2.5))

    start_geom = route["start_point"]
    end_geom = route["end_point"]
    ax.scatter([start_geom.x], [start_geom.y], s=55, zorder=4, label="start")
    ax.scatter([end_geom.x], [end_geom.y], s=55, zorder=4, label="end")

    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.legend(loc="upper right")
    ax.set_title(f"Shortest route: {route.get('route_cost_m', route['route_length_m']):.1f} m")
    return fig, ax, route
