import time
import pickle
from pathlib import Path
from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import pandas as pd
import pybdshadow
from shapely.geometry import LineString, Point, mapping
from shapely import make_valid
from rasterio.transform import from_bounds
from rasterio.features import rasterize as rio_rasterize


DEFAULT_NETWORK_NAME = "toronto"
DEFAULT_WALKING_PATHS_DIR = Path("data/walking-paths")
DEFAULT_BUILDINGS_PATH = Path("data/buildings/buildings.gpkg")
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

    return best_key, best_attrs, best_geom, float(best_attrs.get("length", best_attrs.get(weight_field, 0.0))), best_cost


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


def _projected_route_corridor(start_point: Point, end_point: Point, buffer_m: float) -> LineString:
    return LineString([start_point, end_point]).buffer(buffer_m)


def _clean_geometry(geometry: object) -> object:
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry

    try:
        return geometry if geometry.is_valid else make_valid(geometry)
    except Exception:
        return geometry.buffer(0)

def _rasterize_shadow_union(
    shadow_union,
    crs: str,
    resolution_m: float = 1.0,
) -> tuple[np.ndarray, object]:
    """Burn shadow polygons into a boolean raster. Returns (grid, transform)."""

    minx, miny, maxx, maxy = shadow_union.bounds
    width  = max(1, int((maxx - minx) / resolution_m))
    height = max(1, int((maxy - miny) / resolution_m))

    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    grid = rio_rasterize(
        [(mapping(shadow_union), 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return grid, transform


def _shadow_fraction_from_raster(
    edge_geom,
    grid: np.ndarray,
    transform,
    n_samples: int = 20,
) -> float:
    """Sample edge geometry against raster; return fraction in shadow."""
    if edge_geom is None or edge_geom.is_empty:
        return 0.0

    coords = [
        edge_geom.interpolate(t, normalized=True)
        for t in np.linspace(0, 1, n_samples)
    ]
    # Convert world coords → pixel coords
    xs = np.array([p.x for p in coords])
    ys = np.array([p.y for p in coords])

    # rasterio transform: col = (x - left) / res_x, row = (top - y) / res_y
    res_x = transform.a
    res_y = -transform.e
    cols = ((xs - transform.c) / res_x).astype(int)
    rows = ((transform.f - ys) / res_y).astype(int)

    h, w = grid.shape
    valid = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    if not valid.any():
        return 0.0

    return float(grid[rows[valid], cols[valid]].mean())

def _narrow_corridor_subgraph(
    graph: nx.Graph,
    edges_gdf: gpd.GeoDataFrame,
    routing_index: dict,
    fastest_route: dict,
    narrow_buffer_m: float = 100.0,
) -> tuple[nx.MultiDiGraph, dict]:
    """Build a subgraph restricted to a narrow corridor around the fastest route."""
    route_geometries = [
        e["geometry"] for e in fastest_route["route_edges"]
        if e.get("geometry") is not None
    ]
    if not route_geometries:
        return graph.copy(), {}

    route_line = gpd.GeoSeries(route_geometries, crs=edges_gdf.crs).union_all()
    corridor = route_line.buffer(narrow_buffer_m)

    candidate_idx = list(routing_index["edge_sindex"].query(corridor))
    candidate_edges = edges_gdf.iloc[candidate_idx]

    route_edges = []
    edge_lookup = {}
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

def _annotate_edges_with_shadow(
    route_graph: nx.MultiDiGraph,
    edge_lookup: dict,
    grid: np.ndarray,
    transform,
    weight_field: str,
    shade_weight: float,
) -> None:
    for u, v, key, attrs in route_graph.edges(keys=True, data=True):
        geom = edge_lookup.get((u, v, key)) or edge_lookup.get((v, u, key))
        frac = _shadow_fraction_from_raster(geom, grid, transform) if geom else 0.0
        base = float(attrs.get(weight_field, attrs.get("length", 0.0)))
        attrs["shaded_cost"] = base * (1.0 + shade_weight * (1.0 - frac))

def _route_shadow_fraction(route: dict[str, object], shadow_union: object | None) -> float:
    if shadow_union is None or route.get("route_length_m", 0.0) <= 0:
        return 0.0

    shadow_length_m = 0.0
    for edge in route["route_edges"]:
        geom = edge.get("geometry")
        if geom is None or geom.is_empty:
            continue
        shadow_length_m += geom.intersection(shadow_union).length

    return float(100.0 * shadow_length_m / route["route_length_m"])


def _shadow_penalty_fn_raster(
    grid: np.ndarray,
    transform,
    weight_field: str,
    shade_weight: float,
) -> ShadowPenaltyFn:
    def penalty(edge_geom, attrs: dict) -> float:
        base_cost = float(attrs.get(weight_field, attrs.get("length", 0.0)))
        if base_cost <= 0:
            return 0.0
        shade_frac = _shadow_fraction_from_raster(edge_geom, grid, transform)
        return base_cost * shade_weight * shade_frac
    return penalty

def _compute_shadows_chunk(args):
    buildings_chunk, shadow_time = args
    return pybdshadow.bdshadow_sunlight(buildings_chunk, shadow_time, roof=False, include_building=False)

def _load_local_buildings_for_route(
    buildings_path: str | Path,
    corridor: LineString,
    route_crs: str,
    shadow_time: pd.Timestamp,
    read_buffer_m: float = 200.0,
    source_crs: str = "EPSG:4326",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    search_area = corridor.buffer(read_buffer_m)
    search_area_wgs84 = gpd.GeoSeries([search_area], crs=route_crs).to_crs(source_crs).iloc[0]
    minx, miny, maxx, maxy = search_area_wgs84.bounds

    buildings = gpd.read_file(buildings_path, bbox=(minx, miny, maxx, maxy))
    if buildings.empty:
        return buildings, buildings

    if buildings.crs is None:
        buildings = buildings.set_crs(source_crs)
    else:
        buildings = buildings.to_crs(source_crs)

    if "height" not in buildings.columns:
        height_columns = [column for column in ("MAX_HEIGHT", "AVG_HEIGHT", "MIN_HEIGHT") if column in buildings.columns]
        if height_columns:
            buildings["height"] = buildings[height_columns].max(axis=1)
        else:
            buildings["height"] = 0

    buildings = buildings[buildings["height"].fillna(0) > 0].copy()
    if buildings.empty:
        return buildings, buildings

    buildings = pybdshadow.bd_preprocess(buildings, height="height")

    n_workers = 8
    chunks = np.array_split(buildings, n_workers)
    chunks = [c for c in chunks if not c.empty]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        shadow_parts = list(executor.map(_compute_shadows_chunk, [(c, shadow_time) for c in chunks]))

    shadows = pd.concat(shadow_parts, ignore_index=True)
    shadows = gpd.GeoDataFrame(shadows, crs=buildings.crs)

    buildings = buildings.to_crs(route_crs)
    shadows = shadows.to_crs(route_crs)
    invalid_mask = ~shadows.geometry.is_valid
    if invalid_mask.any():
        shadows.loc[invalid_mask, "geometry"] = shadows.loc[invalid_mask, "geometry"].apply(_clean_geometry)
    return buildings, shadows


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
    routing_index: dict[str, Any] | None = None,
) -> dict[str, object]:

    t0 = time.perf_counter()   

    network_crs = str(edges_gdf.crs)
    start_point = _coerce_point(start_coord, input_crs=input_crs, output_crs=network_crs)
    end_point = _coerce_point(end_coord, input_crs=input_crs, output_crs=network_crs)
    corridor = _projected_route_corridor(start_point, end_point, route_buffer_m)

    buildings_gdf, shadows_gdf = _load_local_buildings_for_route(
        buildings_path=buildings_path,
        corridor=corridor,
        route_crs=network_crs,
        shadow_time=shadow_time,
        read_buffer_m=shadow_read_buffer_m,
    )

    print(f"buildings+shadows: {time.perf_counter()-t0:.2f}s")
    t1 = time.perf_counter()

    # Build shadow union (None if no shadows)
    shadow_union = None
    if not shadows_gdf.empty:
        simplified = shadows_gdf.geometry.apply(
            lambda g: g.simplify(2.0, preserve_topology=False)
        )
        shadow_union = gpd.GeoSeries(simplified, crs=shadows_gdf.crs).buffer(0).union_all()
    
    print(f"union+simplify: {time.perf_counter()-t1:.2f}s")
    t2 = time.perf_counter()

    # Fastest route (pure distance, no shadow awareness)
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

    print(f"fastest route: {time.perf_counter()-t2:.2f}s")
    t3 = time.perf_counter()

    # Shaded route
    if shadow_union is not None:
        # Rasterize once
        grid, transform = _rasterize_shadow_union(shadow_union, crs=network_crs, resolution_m=1.0)

        print(f"rasterize: {time.perf_counter()-t3:.2f}s")
        t4 = time.perf_counter()

        # Build narrow corridor subgraph around fastest route
        route_graph, edge_lookup = _narrow_corridor_subgraph(
            graph=graph,
            edges_gdf=edges_gdf,
            routing_index=routing_index,
            fastest_route=fastest_route,
            narrow_buffer_m=100.0,
        )

        # Pre-cache shaded_cost on every edge — no geometry work during A*
        _annotate_edges_with_shadow(
            route_graph, edge_lookup, grid, transform,
            weight_field="length", shade_weight=shade_weight,
        )

        print(f"annotate edges: {time.perf_counter()-t4:.2f}s")
        t5 = time.perf_counter()

        # A* on pre-annotated subgraph: pure float lookups, no penalty_fn
        shaded_route = compute_shortest_route(
            graph=route_graph,          # pre-annotated subgraph, not full graph
            nodes_gdf=nodes_gdf,
            edges_gdf=edges_gdf,
            start_coord=start_coord,
            end_coord=end_coord,
            input_crs=input_crs,
            weight_field="shaded_cost", # key: route on cached costs
            routing_index=routing_index,
            route_buffer_m=route_buffer_m,
            shadow_penalty_fn=None,     # key: no per-evaluation geometry work
        )
        print(f"shaded route: {time.perf_counter()-t5:.2f}s")
    else:
        shaded_route = fastest_route

    return {
        "shadow_time": shadow_time,
        "buildings_gdf": buildings_gdf,
        "shadows_gdf": shadows_gdf,
        "shadow_union": shadow_union,
        "fastest_route": fastest_route,
        "shaded_route": shaded_route,
        "fastest_shade_pct": _route_shadow_fraction(fastest_route, shadow_union),
        "shaded_shade_pct": _route_shadow_fraction(shaded_route, shadow_union),
    }


def plot_shadow_route_report(report: dict[str, object], ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    buildings_gdf = report["buildings_gdf"]
    shadows_gdf = report["shadows_gdf"]
    fastest_route = report["fastest_route"]
    shaded_route = report["shaded_route"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig = ax.figure

    if isinstance(buildings_gdf, gpd.GeoDataFrame) and not buildings_gdf.empty:
        buildings_gdf.plot(ax=ax, color="#c7c7c7", edgecolor="none", alpha=0.35, zorder=1)
    if isinstance(shadows_gdf, gpd.GeoDataFrame) and not shadows_gdf.empty:
        shadows_gdf.plot(ax=ax, color="#1d4ed8", edgecolor="none", alpha=0.22, zorder=2)

    def _route_segments(route: dict[str, object]) -> list[np.ndarray]:
        return [np.asarray(edge["geometry"].coords) for edge in route["route_edges"] if edge.get("geometry") is not None and hasattr(edge["geometry"], "coords")]

    ax.add_collection(LineCollection(_route_segments(fastest_route), linewidths=2.0, colors="#111827", alpha=0.8, zorder=4))
    ax.add_collection(LineCollection(_route_segments(shaded_route), linewidths=3.0, colors="#dc2626", alpha=0.95, zorder=5))

    ax.scatter([fastest_route["start_point"].x], [fastest_route["start_point"].y], s=55, zorder=6, label="start")
    ax.scatter([fastest_route["end_point"].x], [fastest_route["end_point"].y], s=55, zorder=6, label="end")

    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.legend(loc="upper right")
    ax.set_title(
        f"Fastest: {fastest_route['route_length_m']:.1f} m | Shade-aware: {shaded_route['route_length_m']:.1f} m"
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
