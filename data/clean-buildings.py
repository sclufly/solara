import geopandas as gpd
import pybdshadow

buildings = gpd.read_file("./3DMassingShapefile_2025_WGS84/3DMassingShapefile_2025_WGS84.shp")

buildings = buildings.to_crs(epsg=4326)
buildings["height"] = buildings[["MAX_HEIGHT", "AVG_HEIGHT"]].max(axis=1)
buildings = pybdshadow.bd_preprocess(buildings)

buildings.to_file(
    "./buildings/buildings.gpkg",
    driver="GPKG"
)