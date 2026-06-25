import pandas as pd
import geopandas as gpd
import pybdshadow
import matplotlib.pyplot as plt

buildings = gpd.read_file(r"./data/buildings/buildings_preprocessed.gpkg")

# select sample cluster of buildings
cluster_size = 100
seed_building = buildings.sample(1, random_state=0)
seed_centroid = seed_building.geometry.iloc[0].centroid
centroids = buildings.geometry.centroid
nearest_idx = centroids.distance(seed_centroid).nsmallest(cluster_size).index
buildings = buildings.loc[nearest_idx]

# pre-process buildings
# buildings = buildings.to_crs(epsg=4326)
# buildings["height"] = buildings["MAX_HEIGHT"]
# buildings = pybdshadow.bd_preprocess(buildings)

print(buildings.geometry.name)
print(buildings.head(5))

# create the shadows
date = pd.to_datetime('2022-06-01 12:45:33.959797119')\
    .tz_localize('America/Toronto')

shadows = pybdshadow.bdshadow_sunlight(
    buildings,
    date,
    roof=True,
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