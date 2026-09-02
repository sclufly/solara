from pathlib import Path
from typing import Callable

BACKEND_DIR = Path(__file__).resolve().parent.parent

DEFAULT_NETWORK_NAME = "toronto"
DEFAULT_WALKING_PATHS_DIR = BACKEND_DIR / "data" / "walking-paths"
DEFAULT_BUILDINGS_PATH = BACKEND_DIR / "data" / "buildings" / "buildings.gpkg"

ShadowPenaltyFn = Callable[[object, dict[str, object]], float]
