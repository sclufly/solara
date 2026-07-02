import time

import pandas as pd
import matplotlib.pyplot as plt

from routing import (
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

    # cn tower
    start_coord = (-79.385882, 43.642017)
    # eaton center
    end_coord = (-79.379964, 43.652288)
    # cbc building 
    #end_coord = (-79.388826, 43.644417)

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

    fig, ax = plot_shadow_route_report(report)

    stats = [
        f"Shadow time: {shadow_time.strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        f"Fastest route: {report['fastest_route']['route_length_m']:.1f} m  |  "
        f"{len(report['fastest_route']['route_nodes'])} nodes  |  "
        f"{report['fastest_shade_pct']:.1f}% shade",
        "",
        f"Shade-aware route: {report['shaded_route']['route_length_m']:.1f} m  |  "
        f"{len(report['shaded_route']['route_nodes'])} nodes  |  "
        f"{report['shaded_shade_pct']:.1f}% shade",
    ]

    ax.text(
        0.01, 0.01,
        "\n".join(stats),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    plt.show()


if __name__ == "__main__":
    main()