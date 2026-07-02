"""Building shadow computation, rasterization, and edge annotation."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pybdshadow
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds
from shapely.geometry import LineString, mapping
from shapely import make_valid


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _clean_geometry(geometry: object) -> object:
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    try:
        return geometry if geometry.is_valid else make_valid(geometry)
    except Exception:
        return geometry.buffer(0)


# ---------------------------------------------------------------------------
# Building + shadow loading
# ---------------------------------------------------------------------------

def _compute_shadows_chunk(args: tuple) -> gpd.GeoDataFrame:
    buildings_chunk, shadow_time = args
    return pybdshadow.bdshadow_sunlight(buildings_chunk, shadow_time, roof=False, include_building=False)


def load_buildings_and_shadows(
    buildings_path: str | Path,
    corridor: LineString,
    route_crs: str,
    shadow_time: pd.Timestamp,
    read_buffer_m: float = 200.0,
    source_crs: str = "EPSG:4326",
    n_workers: int = 8,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load buildings within a buffered corridor and compute their shadows."""
    search_area      = corridor.buffer(read_buffer_m)
    search_area_wgs84 = gpd.GeoSeries([search_area], crs=route_crs).to_crs(source_crs).iloc[0]
    minx, miny, maxx, maxy = search_area_wgs84.bounds

    buildings = gpd.read_file(buildings_path, bbox=(minx, miny, maxx, maxy))
    if buildings.empty:
        return buildings, buildings

    buildings = buildings.set_crs(source_crs) if buildings.crs is None else buildings.to_crs(source_crs)
    buildings = _ensure_height_column(buildings)
    buildings = buildings[buildings["height"].fillna(0) > 0].copy()

    if buildings.empty:
        return buildings, buildings

    buildings = pybdshadow.bd_preprocess(buildings, height="height")
    shadows   = _compute_shadows_parallel(buildings, shadow_time, n_workers)

    buildings = buildings.to_crs(route_crs)
    shadows   = shadows.to_crs(route_crs)

    invalid_mask = ~shadows.geometry.is_valid
    if invalid_mask.any():
        shadows.loc[invalid_mask, "geometry"] = shadows.loc[invalid_mask, "geometry"].apply(_clean_geometry)

    return buildings, shadows


def _ensure_height_column(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "height" in buildings.columns:
        return buildings
    height_cols = [c for c in ("MAX_HEIGHT", "AVG_HEIGHT", "MIN_HEIGHT") if c in buildings.columns]
    buildings["height"] = buildings[height_cols].max(axis=1) if height_cols else 0
    return buildings


def _compute_shadows_parallel(
    buildings: gpd.GeoDataFrame,
    shadow_time: pd.Timestamp,
    n_workers: int,
) -> gpd.GeoDataFrame:
    chunks = [c for c in np.array_split(buildings, n_workers) if not c.empty]
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        parts = list(executor.map(_compute_shadows_chunk, [(c, shadow_time) for c in chunks]))
    shadows = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(shadows, crs=buildings.crs)


# ---------------------------------------------------------------------------
# Shadow union
# ---------------------------------------------------------------------------

def build_shadow_union(shadows_gdf: gpd.GeoDataFrame, simplify_tolerance_m: float = 2.0) -> object | None:
    """Merge shadow polygons into a single geometry, simplified for performance."""
    if shadows_gdf.empty:
        return None
    simplified = shadows_gdf.geometry.apply(lambda g: g.simplify(simplify_tolerance_m, preserve_topology=False))
    return gpd.GeoSeries(simplified, crs=shadows_gdf.crs).buffer(0).union_all()


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def rasterize_shadow_union(
    shadow_union,
    resolution_m: float = 1.0,
) -> tuple[np.ndarray, object]:
    """Burn the shadow union into a boolean uint8 raster. Returns (grid, transform)."""
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


def shadow_fraction_from_raster(
    edge_geom,
    grid: np.ndarray,
    transform,
    n_samples: int = 20,
) -> float:
    """Sample an edge geometry against the shadow raster; return shaded fraction [0, 1]."""
    if edge_geom is None or edge_geom.is_empty:
        return 0.0

    ts   = np.linspace(0, 1, n_samples)
    pts  = [edge_geom.interpolate(t, normalized=True) for t in ts]
    xs   = np.array([p.x for p in pts])
    ys   = np.array([p.y for p in pts])

    res_x = transform.a
    res_y = -transform.e
    cols  = ((xs - transform.c) / res_x).astype(int)
    rows  = ((transform.f - ys) / res_y).astype(int)

    h, w  = grid.shape
    valid = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    return float(grid[rows[valid], cols[valid]].mean()) if valid.any() else 0.0


# ---------------------------------------------------------------------------
# Edge annotation
# ---------------------------------------------------------------------------

def annotate_edges_with_shadow(
    route_graph: nx.MultiDiGraph,
    edge_lookup: dict,
    grid: np.ndarray,
    transform,
    weight_field: str,
    shade_weight: float,
) -> None:
    """Pre-cache shaded_cost on every edge in-place so A* needs no geometry calls."""
    for u, v, key, attrs in route_graph.edges(keys=True, data=True):
        geom = edge_lookup.get((u, v, key)) or edge_lookup.get((v, u, key))
        frac = shadow_fraction_from_raster(geom, grid, transform) if geom else 0.0
        base = float(attrs.get(weight_field, attrs.get("length", 0.0)))
        attrs["shaded_cost"] = base * (1.0 + shade_weight * (1.0 - frac))


# ---------------------------------------------------------------------------
# Route shadow reporting
# ---------------------------------------------------------------------------

def route_shadow_fraction(route: dict[str, object], shadow_union: object | None) -> float:
    """Return the percentage of a route's physical length that falls in shadow."""
    if shadow_union is None or route.get("route_length_m", 0.0) <= 0:
        return 0.0

    shadow_length_m = sum(
        edge["geometry"].intersection(shadow_union).length
        for edge in route["route_edges"]
        if edge.get("geometry") is not None and not edge["geometry"].is_empty
    )
    return float(100.0 * shadow_length_m / route["route_length_m"])
