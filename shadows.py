import pandas as pd
import geopandas as gpd
import pybdshadow
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = r"./data/buildings/buildings.gpkg"

# select a cluster of buildings within a local area of interest (AOI)
def select_cluster_from_aoi(data_path, cluster_size=500, initial_radius_m=1200, max_radius_m=20000):

    # select a seed building
    seed_building = gpd.read_file(data_path, rows=slice(138194, 138195))
    
    seed_centroid = seed_building.geometry.iloc[0].centroid
    seed_x, seed_y = seed_centroid.x, seed_centroid.y
    radius_m = initial_radius_m
    buildings = gpd.GeoDataFrame()

    # expand AOI until we have enough candidates
    while radius_m <= max_radius_m:
        # approx meter-to-degree conversion for WGS84 readtime bbox
        dlat = radius_m / 111320.0
        dlon = radius_m / (111320.0 * max(0.1, abs(np.cos(np.radians(seed_y)))))
        bbox = (seed_x - dlon, seed_y - dlat, seed_x + dlon, seed_y + dlat)

        buildings = gpd.read_file(data_path, bbox=bbox)
        if len(buildings) >= cluster_size:
            break

        radius_m *= 2

    if len(buildings) == 0:
        raise ValueError("AOI query returned no buildings")

    # project only local AOI and use vectorized centroid distance in meters
    metric_crs = buildings.estimate_utm_crs()
    buildings_metric = buildings.to_crs(metric_crs)
    seed_metric = gpd.GeoSeries([seed_centroid], crs=seed_building.crs).to_crs(metric_crs).iloc[0]

    centroids = buildings_metric.geometry.centroid
    dx = centroids.x - seed_metric.x
    dy = centroids.y - seed_metric.y
    buildings_metric["dist2"] = dx * dx + dy * dy

    cluster = buildings_metric.nsmallest(min(cluster_size, len(buildings_metric)), "dist2")
    cluster = cluster.drop(columns=["dist2"]).to_crs(buildings.crs)

    return cluster

# select sample cluster of buildings with AOI query
cluster_size = 2500
buildings = select_cluster_from_aoi(DATA_PATH, cluster_size=cluster_size)

print("== Cluster selection complete, creating shadows... ==")

# create the shadows
date = pd.to_datetime('2022-11-01 16:45:33.959797119').tz_localize('America/Toronto')

shadows = pybdshadow.bdshadow_sunlight(
    buildings,
    date,
    roof=False,
    include_building=False
)

# plot the shapes
fig = plt.figure(1, (12, 12))
ax = plt.subplot(111)

buildings.plot(ax=ax)

shadows.plot(ax=ax, alpha=0.7,
             column='type',
             categorical=True,
             cmap='Set1_r',
             legend=True)

plt.show()