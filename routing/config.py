from pathlib import Path
from typing import Callable

DEFAULT_NETWORK_NAME = "toronto"
DEFAULT_WALKING_PATHS_DIR = Path("data/walking-paths")
DEFAULT_BUILDINGS_PATH = Path("data/buildings/buildings.gpkg")

ShadowPenaltyFn = Callable[[object, dict[str, object]], float]
