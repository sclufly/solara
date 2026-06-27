from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


DEFAULT_SHAPEFILE = Path(
    r"data/Pedestrian Network Data - 4326/Pedestrian Network Data - 4326.shp"
)

DEFAULT_GRAPH_CACHE_DIR = Path(r"data/cache/pedestrian_graph")

SOURCE_MATCH_TOLERANCE_M = 0.25


def load_projected_pedestrian_network(
    shapefile_path: str | Path,
    target_crs: str | None = None,
) -> gpd.GeoDataFrame:
    """Load the pedestrian network and project it into a metric CRS.

    The raw shapefile is stored in EPSG:4326. For graph construction, all
    distances should be measured in meters, so we project the data before
    building edges and heuristic-friendly node coordinates.
    """

    gdf = gpd.read_file(shapefile_path)
    if gdf.empty:
        raise ValueError(f"No features were read from {shapefile_path}")

    if gdf.crs is None:
        raise ValueError("The pedestrian network shapefile has no CRS defined")

    if gdf.crs.is_projected:
        projected = gdf.copy()
    else:
        metric_crs = target_crs or gdf.estimate_utm_crs()
        projected = gdf.to_crs(metric_crs)

    projected = projected.reset_index(drop=True).copy()
    projected["source_feature_id"] = projected.index.astype(int)
    return projected


def _iter_line_geometries(geometry: BaseGeometry) -> Iterable[LineString]:
    if geometry.is_empty:
        return

    if isinstance(geometry, LineString):
        yield geometry
        return

    if isinstance(geometry, MultiLineString):
        for part in geometry.geoms:
            yield from _iter_line_geometries(part)
        return

    geom_type = geometry.geom_type
    if geom_type == "GeometryCollection":
        for part in geometry.geoms:
            yield from _iter_line_geometries(part)


def _point_key(point: Point, precision: int) -> tuple[float, float]:
    return (round(float(point.x), precision), round(float(point.y), precision))


def _ensure_node(
    graph: nx.MultiGraph,
    node_lookup: dict[tuple[float, float], int],
    point: Point,
    precision: int,
) -> int:
    key = _point_key(point, precision)
    node_id = node_lookup.get(key)
    if node_id is None:
        node_id = len(node_lookup)
        node_lookup[key] = node_id
        graph.add_node(
            node_id,
            x=float(point.x),
            y=float(point.y),
            geometry=Point(float(point.x), float(point.y)),
            coordinate_key=key,
        )
    return node_id


def _source_feature_ids(
    segment: LineString,
    source_gdf: gpd.GeoDataFrame,
) -> tuple[int, ...]:
    midpoint = segment.interpolate(0.5, normalized=True)
    search_area = midpoint.buffer(SOURCE_MATCH_TOLERANCE_M)
    candidate_indexes = source_gdf.sindex.query(search_area, predicate="intersects")
    matching_ids: list[int] = []

    for candidate_index in candidate_indexes:
        candidate_row = source_gdf.iloc[int(candidate_index)]
        if candidate_row.geometry.distance(midpoint) <= SOURCE_MATCH_TOLERANCE_M:
            matching_ids.append(int(candidate_row.source_feature_id))

    return tuple(sorted(set(matching_ids)))


def build_pedestrian_graph(
    shapefile_path: str | Path = DEFAULT_SHAPEFILE,
    target_crs: str | None = None,
    node_precision: int = 3,
) -> tuple[nx.MultiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build a topology-aware pedestrian graph from the line shapefile.

    The graph is intentionally a MultiGraph so parallel walkable segments are
    preserved instead of being collapsed into a single edge.
    """

    network = load_projected_pedestrian_network(shapefile_path, target_crs=target_crs)

    graph = nx.MultiGraph()
    graph.graph["crs"] = network.crs
    graph.graph["source_shapefile"] = str(shapefile_path)
    graph.graph["node_precision"] = node_precision

    node_lookup: dict[tuple[float, float], int] = {}
    raw_linework = unary_union(network.geometry.to_list())

    edge_rows: list[dict[str, object]] = []
    for edge_id, segment in enumerate(_iter_line_geometries(raw_linework)):
        if segment.is_empty or segment.length <= 0:
            continue

        coords = list(segment.coords)
        if len(coords) < 2:
            continue

        start_point = Point(coords[0])
        end_point = Point(coords[-1])
        u = _ensure_node(graph, node_lookup, start_point, node_precision)
        v = _ensure_node(graph, node_lookup, end_point, node_precision)

        source_ids = _source_feature_ids(segment, network)
        edge_data = {
            "edge_id": edge_id,
            "geometry": segment,
            "length_m": float(segment.length),
            "source_feature_ids": source_ids,
        }
        graph.add_edge(u, v, key=edge_id, **edge_data)

        edge_rows.append(
            {
                "u": u,
                "v": v,
                "key": edge_id,
                **edge_data,
            }
        )

    node_rows = [
        {
            "node_id": node_id,
            "x": data["x"],
            "y": data["y"],
            "geometry": data["geometry"],
            "degree": graph.degree[node_id],
            "coordinate_key": data["coordinate_key"],
        }
        for node_id, data in graph.nodes(data=True)
    ]

    nodes_gdf = gpd.GeoDataFrame(node_rows, geometry="geometry", crs=network.crs)
    edges_gdf = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=network.crs)
    return graph, nodes_gdf, edges_gdf, network


def save_graph_bundle(
    cache_dir: str | Path,
    graph: nx.MultiGraph,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Persist a built graph bundle so it does not need to be rebuilt.

    The graph is saved with pickle, while the node and edge tables are stored as
    GeoParquet for easier inspection and reuse.
    """

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    with (cache_path / "graph.pkl").open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    nodes_gdf.to_parquet(cache_path / "nodes.parquet", index=False)
    edges_gdf.to_parquet(cache_path / "edges.parquet", index=False)

    payload = {
        "crs": str(graph.graph.get("crs")) if graph.graph.get("crs") is not None else None,
        "source_shapefile": graph.graph.get("source_shapefile"),
        "node_precision": graph.graph.get("node_precision"),
    }
    if metadata:
        payload.update(metadata)

    with (cache_path / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    return cache_path


def load_graph_bundle(
    cache_dir: str | Path,
) -> tuple[nx.MultiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, object]]:
    """Load a previously saved graph bundle."""

    cache_path = Path(cache_dir)
    with (cache_path / "graph.pkl").open("rb") as handle:
        graph = pickle.load(handle)

    nodes_gdf = gpd.read_parquet(cache_path / "nodes.parquet")
    edges_gdf = gpd.read_parquet(cache_path / "edges.parquet")

    metadata_path = cache_path / "metadata.json"
    metadata: dict[str, object]
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    else:
        metadata = {}

    return graph, nodes_gdf, edges_gdf, metadata


def load_or_build_graph_bundle(
    cache_dir: str | Path = DEFAULT_GRAPH_CACHE_DIR,
    shapefile_path: str | Path = DEFAULT_SHAPEFILE,
    target_crs: str | None = None,
    node_precision: int = 3,
    force_rebuild: bool = False,
) -> tuple[nx.MultiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, object]]:
    """Load a cached graph bundle when available, otherwise build and cache it."""

    cache_path = Path(cache_dir)
    if not force_rebuild and (cache_path / "graph.pkl").exists():
        return load_graph_bundle(cache_path)

    graph, nodes_gdf, edges_gdf, _network = build_pedestrian_graph(
        shapefile_path=shapefile_path,
        target_crs=target_crs,
        node_precision=node_precision,
    )
    metadata = {
        "shapefile_path": str(shapefile_path),
        "target_crs": target_crs,
    }
    save_graph_bundle(cache_path, graph, nodes_gdf, edges_gdf, metadata=metadata)
    return load_graph_bundle(cache_path)


def nearest_node(
    nodes_gdf: gpd.GeoDataFrame,
    point: Point,
) -> int:
    """Return the closest graph node to a point.

    This is a small helper for future routing steps when start and end points
    are not exactly on a node.
    """

    if nodes_gdf.empty:
        raise ValueError("Cannot snap to a node in an empty graph")

    distances = nodes_gdf.geometry.distance(point)
    return int(nodes_gdf.loc[distances.idxmin(), "node_id"])


def _coerce_point(
    coordinate: tuple[float, float] | list[float] | Point,
    input_crs: str,
    output_crs: str,
) -> Point:
    if isinstance(coordinate, Point):
        point = coordinate
        if input_crs == output_crs:
            return point
    else:
        point = Point(float(coordinate[0]), float(coordinate[1]))

    point_series = gpd.GeoSeries([point], crs=input_crs)
    return point_series.to_crs(output_crs).iloc[0]


def _edge_for_node_pair(
    graph: nx.MultiGraph,
    u: int,
    v: int,
) -> tuple[int, dict[str, object]]:
    edge_data = graph.get_edge_data(u, v)
    if not edge_data:
        raise ValueError(f"No edge exists between nodes {u} and {v}")

    key, attrs = min(edge_data.items(), key=lambda item: float(item[1].get("length_m", float("inf"))))
    return int(key), attrs


def compute_shortest_route(
    graph: nx.MultiGraph,
    nodes_gdf: gpd.GeoDataFrame,
    start_coord: tuple[float, float] | list[float] | Point,
    end_coord: tuple[float, float] | list[float] | Point,
    input_crs: str = "EPSG:4326",
) -> dict[str, object]:
    """Compute a length-based route between two coordinates."""

    graph_crs = str(graph.graph.get("crs"))
    start_point = _coerce_point(start_coord, input_crs=input_crs, output_crs=graph_crs)
    end_point = _coerce_point(end_coord, input_crs=input_crs, output_crs=graph_crs)

    start_node = nearest_node(nodes_gdf, start_point)
    end_node = nearest_node(nodes_gdf, end_point)

    route_nodes = nx.shortest_path(graph, start_node, end_node, weight="length_m")
    route_edges: list[dict[str, object]] = []
    route_length_m = 0.0

    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        key, attrs = _edge_for_node_pair(graph, u, v)
        route_edges.append({"u": u, "v": v, "key": key, **attrs})
        route_length_m += float(attrs["length_m"])

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
    graph: nx.MultiGraph,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    start_coord: tuple[float, float] | list[float] | Point,
    end_coord: tuple[float, float] | list[float] | Point,
    input_crs: str = "EPSG:4326",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, dict[str, object]]:
    """Plot the network and a computed route between two coordinate inputs."""

    route = compute_shortest_route(
        graph=graph,
        nodes_gdf=nodes_gdf,
        start_coord=start_coord,
        end_coord=end_coord,
        input_crs=input_crs,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig = ax.figure

    edges_gdf.plot(ax=ax, color="#c8c8c8", linewidth=0.35, alpha=0.45, zorder=1)

    route_edge_geoms = [edge["geometry"] for edge in route["route_edges"]]
    route_edges_gdf = gpd.GeoSeries(route_edge_geoms, crs=edges_gdf.crs)
    route_edges_gdf.plot(ax=ax, color="#d7301f", linewidth=2.5, alpha=0.95, zorder=3)

    start_geom = route["start_point"]
    end_geom = route["end_point"]
    ax.scatter([start_geom.x], [start_geom.y], s=55, c="#1a9850", edgecolors="black", linewidths=0.6, zorder=4, label="start")
    ax.scatter([end_geom.x], [end_geom.y], s=55, c="#2b83ba", edgecolors="black", linewidths=0.6, zorder=4, label="end")

    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.legend(loc="upper right")
    ax.set_title(f"Shortest route by length: {route['route_length_m']:.1f} m")

    return fig, ax, route


def plot_route_on_original_shapefile(
    graph: nx.MultiGraph,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    original_shapefile_path: str | Path,
    start_coord: tuple[float, float] | list[float] | Point,
    end_coord: tuple[float, float] | list[float] | Point,
    input_crs: str = "EPSG:4326",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, dict[str, object]]:
    """Plot a computed route on top of the original shapefile coordinates."""

    original_network = gpd.read_file(original_shapefile_path)
    if original_network.crs is None:
        raise ValueError("The original shapefile has no CRS defined")

    route = compute_shortest_route(
        graph=graph,
        nodes_gdf=nodes_gdf,
        start_coord=start_coord,
        end_coord=end_coord,
        input_crs=input_crs,
    )

    route_geom = gpd.GeoSeries([edge["geometry"] for edge in route["route_edges"]], crs=graph.graph["crs"])
    route_geom = route_geom.to_crs(original_network.crs)

    start_point = gpd.GeoSeries([route["start_point"]], crs=graph.graph["crs"]).to_crs(original_network.crs).iloc[0]
    end_point = gpd.GeoSeries([route["end_point"]], crs=graph.graph["crs"]).to_crs(original_network.crs).iloc[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig = ax.figure

    original_network.plot(ax=ax, color="#8f8f8f", linewidth=0.6, alpha=0.6, zorder=1)
    route_geom.plot(ax=ax, color="#d7301f", linewidth=2.8, alpha=0.95, zorder=3)

    ax.scatter([start_point.x], [start_point.y], s=60, c="#1a9850", edgecolors="black", linewidths=0.6, zorder=4, label="start")
    ax.scatter([end_point.x], [end_point.y], s=60, c="#2b83ba", edgecolors="black", linewidths=0.6, zorder=4, label="end")

    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.legend(loc="upper right")
    ax.set_title(f"Route on original shapefile: {route['route_length_m']:.1f} m")

    return fig, ax, route


def summarize_graph(graph: nx.MultiGraph) -> str:
    return f"nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}, crs={graph.graph.get('crs')}"


if __name__ == "__main__":
    graph, nodes_gdf, edges_gdf, metadata = load_or_build_graph_bundle()
    print(summarize_graph(graph))
    print(f"nodes table rows={len(nodes_gdf)}")
    print(f"edges table rows={len(edges_gdf)}")
    print(f"cache metadata keys={sorted(metadata.keys())}")

    start_coord = (43.642017, -79.385882)
    end_coord = (43.644417, -79.388826)
    start_lonlat = (start_coord[1], start_coord[0])
    end_lonlat = (end_coord[1], end_coord[0])

    fig, ax, route = plot_route_on_original_shapefile(
        graph=graph,
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        original_shapefile_path=DEFAULT_SHAPEFILE,
        start_coord=start_lonlat,
        end_coord=end_lonlat,
        input_crs="EPSG:4326",
    )
    print(f"route nodes={len(route['route_nodes'])}")
    print(f"route length m={route['route_length_m']:.1f}")
    plt.show()