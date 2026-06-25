import geopandas as gpd
import pybdshadow

buildings = gpd.read_file("./data/buildings/buildings.shp")

buildings = buildings.to_crs(epsg=4326)
buildings["height"] = buildings["MAX_HEIGHT"]
buildings = pybdshadow.bd_preprocess(buildings)

buildings.to_file(
    "./data/buildings/buildings_preprocessed.gpkg",
    driver="GPKG"
)