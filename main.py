import time
import matplotlib.pyplot as plt
from pedestrian_network_graph import DEFAULT_NETWORK_NAME, TorontoRoutingEngine, plot_route_between_coordinates


def main() -> None:
    t0 = time.perf_counter()

    engine = TorontoRoutingEngine.load(network_name=DEFAULT_NETWORK_NAME)

    print(f"nodes table rows={len(engine.nodes_gdf)}")
    print(f"edges table rows={len(engine.edges_gdf)}")
    print(f"dataset={engine.metadata['network_name']}")
    print(f"weight field={engine.metadata['weight_field']}")
    print(f"bundle format={engine.metadata['bundle_format']}")

    t1 = time.perf_counter()
    print("=> load graph bundle:", t1 - t0)

    start_coord = (-79.385882, 43.642017)
    end_coord = (-79.388826, 43.644417)

    fig, ax, route = plot_route_between_coordinates(
        graph=engine.graph,
        nodes_gdf=engine.nodes_gdf,
        edges_gdf=engine.edges_gdf,
        start_coord=start_coord,
        end_coord=end_coord,
        input_crs="EPSG:4326",
        weight_field=str(engine.metadata["weight_field"]),
        routing_index=engine.routing_index,
        full_network=False,
        route_buffer_m=150.0,
    )

    t2 = time.perf_counter()
    print("=> routing:", t2 - t1)

    print(f"route nodes={len(route['route_nodes'])}")
    print(f"route length m={route['route_length_m']:.1f}")
    plt.show()


if __name__ == "__main__":
    main()