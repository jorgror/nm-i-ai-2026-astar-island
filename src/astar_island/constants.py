"""Core constants for Astar Island local tooling."""

from __future__ import annotations

NUM_CLASSES = 6
DEFAULT_MAP_WIDTH = 40
DEFAULT_MAP_HEIGHT = 40
DEFAULT_SEEDS_COUNT = 5

GRID_TO_CLASS = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    10: 0,
    11: 0,
}

CLASS_NAMES = {
    0: "empty",
    1: "settlement",
    2: "port",
    3: "ruin",
    4: "forest",
    5: "mountain",
}

GRID_COLORS = {
    0: "#d9d9d9",   # empty
    1: "#f4a261",   # settlement
    2: "#2a9d8f",   # port
    3: "#7f5539",   # ruin
    4: "#5aa469",   # forest
    5: "#6c757d",   # mountain
    10: "#5dade2",  # ocean
    11: "#f1e3b0",  # plains
}

CLASS_COLORS = {
    0: "#d9d9d9",
    1: "#f4a261",
    2: "#2a9d8f",
    3: "#7f5539",
    4: "#5aa469",
    5: "#6c757d",
}
