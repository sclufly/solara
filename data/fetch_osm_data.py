import json
from pathlib import Path

import osmnx as ox

G = ox.graph_from_place("Toronto, Ontario, Canada", network_type="walk")

ox.plot_graph(G)

G = ox.project_graph(G)

output_dir = Path("./walking-paths")
output_dir.mkdir(parents=True, exist_ok=True)

ox.save_graphml(G, filepath=output_dir / "toronto-walking-paths.graphml")

nodes, edges = ox.graph_to_gdfs(G)


def _json_safe_value(value):
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, ensure_ascii=True)
    return value


nodes = nodes.copy()
edges = edges.copy()

for frame in (nodes, edges):
    for column in frame.columns:
        if column == frame.geometry.name:
            continue
        frame[column] = frame[column].map(_json_safe_value)

nodes.to_file(output_dir / "toronto-network.gpkg", layer="nodes", driver="GPKG")
edges.to_file(output_dir / "toronto-network.gpkg", layer="edges", driver="GPKG")

print(f"Saved GraphML to {output_dir / 'toronto-walking-paths.graphml'}")
print(f"Saved GeoPackage layers to {output_dir / 'toronto-network.gpkg'}")