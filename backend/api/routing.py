from turtle import pd

from backend.main import coordinate

from backend.routing import engine


def compute_route(start_coord: coordinate, end_coord: coordinate):
       report = engine.shadow_route_report(
             start_coord=(start_coord.long, start_coord.lat),
             end_coord=(end_coord.long, end_coord.lat),
             shadow_time=pd.Timestamp(now(), tz="America/Toronto")