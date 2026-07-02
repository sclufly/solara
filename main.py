import time

import pandas as pd
import matplotlib.pyplot as plt

from pedestrian_network_graph import (
    DEFAULT_BUILDINGS_PATH,
    DEFAULT_NETWORK_NAME,
    TorontoRoutingEngine,
    compute_shadow_aware_route_report,
    plot_shadow_route_report,
)


def main() -> None:
    t0 = time.perf_counter()

    engine = TorontoRoutingEngine.load(network_name=DEFAULT_NETWORK_NAME)
    shadow_time = pd.Timestamp.now(tz="America/Toronto")

    print(f"nodes table rows={len(engine.nodes_gdf)}")
    print(f"edges table rows={len(engine.edges_gdf)}")
    print(f"dataset={engine.metadata['network_name']}")
    print(f"weight field={engine.metadata['weight_field']}")
    print(f"bundle format={engine.metadata['bundle_format']}")

    t1 = time.perf_counter()
    print("=> load graph bundle:", t1 - t0)

    start_coord = (-79.385882, 43.642017)
    end_coord = (-79.388826, 43.644417)

    report = compute_shadow_aware_route_report(
        graph=engine.graph,
        nodes_gdf=engine.nodes_gdf,
        edges_gdf=engine.edges_gdf,
        start_coord=start_coord,
        end_coord=end_coord,
        input_crs="EPSG:4326",
        shadow_time=shadow_time,
        buildings_path=DEFAULT_BUILDINGS_PATH,
        routing_index=engine.routing_index,
        route_buffer_m=250.0,
        shadow_read_buffer_m=600.0,
        shade_weight=0.75,
    )

    t2 = time.perf_counter()
    print("=> routing + shadows:", t2 - t1)
    print(f"shadow time={shadow_time.isoformat()}")
    print("fastest route = baseline shortest path")
    print(f"fastest route nodes={len(report['fastest_route']['route_nodes'])}")
    print(f"fastest route length m={report['fastest_route']['route_length_m']:.1f}")
    print(f"fastest route shade cover %={report['fastest_shade_pct']:.1f}")
    print("shade-aware route = shortest path biased toward shadow")
    print(f"shade-aware route nodes={len(report['shaded_route']['route_nodes'])}")
    print(f"shade-aware route length m={report['shaded_route']['route_length_m']:.1f}")
    print(f"shade-aware route shade cover %={report['shaded_shade_pct']:.1f}")

    fig, ax = plot_shadow_route_report(report)
    plt.show()


if __name__ == "__main__":
    main()