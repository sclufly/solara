# SolarA*

SolarA* is a geospatial routing system for pedestrian navigation built on top of OpenStreetMap and City of Toronto open data. The goal of the project is to go beyond standard shortest-path routing by introducing environmental awareness, specifically sunlight and shadow exposure, into route selection.

The system uses a graph-based representation of the city and supports both traditional A* routing and an extended version that incorporates shadow penalties derived from building geometry and sun position.

## What it does

- __Builds pedestrian network graphs from OpenStreetMap.__  
  The city is represented as a graph where nodes are intersections and edges are walkable street segments. This allows classical graph search algorithms to be applied to real-world geography.

- __Computes shortest paths using A* search.__  
  Routes are calculated using the A* algorithm, which efficiently finds the lowest-cost path between two points by combining actual travel cost with a heuristic estimate of remaining distance.

- __Uses City of Toronto building footprint data.__  
  Building geometry and height information is used to simulate shadow casting at a given time of day (via `pybdshadow`), enabling time-dependent environmental routing.

- __Adjusts routing cost using shadow exposure.__  
  Instead of treating all walking segments equally, the system can decrease the cost of edges that fall under building shadows. This allows the model to prefer more comfortable walking routes.

## Inputs

- __OpenStreetMap pedestrian network.__  
  Provides the base street graph including walkable paths, intersections, and connectivity.

- __City of Toronto building footprints.__  
  Provides building geometry and height attributes used to compute shadow casting. This enables environmental context to be added to the routing model.

- __Start and End point coordinates.__  
  Locations for the start and end of the calculated path.

## Outputs

- __Visualization of routes + network + shadows.__  
  Both the fastest route (pure distance optimization) and a shadow-aware route (distance + shadow tradeoff) are displayed, including the total distance of each route and the % of route that's in the shade. The underlying street network, buildings, and shadows are also shown.
