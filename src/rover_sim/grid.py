"""The rover's world: cell types and grid generation.

Design choice: cell coordinates are (row, col), matching numpy's own
indexing convention (grid[row, col]). That's the convention used
everywhere in this file and later in pathfinding.py. The (x, y)
convention from the spec only shows up later, at the boundary where the
environment builds the observation dict handed to the agent.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class CellType(IntEnum):
    """What a single grid cell contains.

    IntEnum, not plain Enum: the grid is stored as a numpy array of plain
    integers for speed (numpy doesn't know about Python enums), but the
    rest of the code can still write the readable CellType.OBSTACLE
    instead of a magic number like 1.
    """

    FREE = 0
    OBSTACLE = 1
    RESOURCE = 2
    HAZARD = 3
    BASE = 4
    UNKNOWN = 5  # only appears in an agent's belief map, never in the true grid


Coord = tuple[int, int]  # (row, col)


def neighbors4(pos: Coord, height: int, width: int) -> list[Coord]:
    """The up/down/left/right neighbors of pos that are still inside the grid."""
    row, col = pos
    candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
    return [(r, c) for r, c in candidates if 0 <= r < height and 0 <= c < width]


def bfs_reachable(grid: np.ndarray, start: Coord) -> set[Coord]:
    """All cells reachable from start by moving through non-obstacle cells.

    Breadth-first search: visit start, then all of its neighbors, then all
    of *their* unvisited neighbors, and so on -- expanding outward one
    "ring" at a time until nothing new is found. Used here to validate a
    freshly generated map, and again, unmodified, as the return-to-base
    planner over the agent's belief map (Milestone 5).
    """
    height, width = grid.shape
    visited: set[Coord] = {start}
    frontier = [start]
    while frontier:
        next_frontier: list[Coord] = []
        for pos in frontier:
            for n in neighbors4(pos, height, width):
                if n not in visited and grid[n] != CellType.OBSTACLE:
                    visited.add(n)
                    next_frontier.append(n)
        frontier = next_frontier
    return visited


def generate_grid(
    height: int = 12,
    width: int = 16,
    seed: int = 0,
    obstacle_density: float = 0.2,
    num_resources: int = 5,
    num_hazards: int = 4,
    max_attempts: int = 50,
) -> tuple[np.ndarray, Coord]:
    """Build a random but reproducible grid, guaranteed fully playable.

    Reproducible: a numpy Generator seeded with `seed` produces the exact
    same sequence of "random" draws every time it's given that seed --
    this is what lets "same seed -> identical run" hold for the whole
    project.

    Guaranteed playable: obstacles are placed randomly, which can
    accidentally wall off part of the map from the base. After placing
    everything, bfs_reachable checks that every resource is reachable; if
    a layout stranded one, we throw it away and generate a fresh layout
    from the same evolving random stream, up to max_attempts times.
    """
    rng = np.random.default_rng(seed)

    for _ in range(max_attempts):
        grid = np.full((height, width), CellType.FREE, dtype=np.int8)

        obstacle_mask = rng.random((height, width)) < obstacle_density
        grid[obstacle_mask] = CellType.OBSTACLE

        free_cells = [(int(r), int(c)) for r, c in zip(*np.where(grid == CellType.FREE))]
        rng.shuffle(free_cells)

        needed = 1 + num_resources + num_hazards
        if len(free_cells) < needed:
            continue  # this layout has too few free cells, try another

        base_pos = free_cells[0]
        resource_cells = free_cells[1 : 1 + num_resources]
        hazard_cells = free_cells[1 + num_resources : needed]

        grid[base_pos] = CellType.BASE
        for pos in resource_cells:
            grid[pos] = CellType.RESOURCE
        for pos in hazard_cells:
            grid[pos] = CellType.HAZARD

        reachable = bfs_reachable(grid, base_pos)
        if all(pos in reachable for pos in resource_cells):
            return grid, base_pos
        # else: a resource got stranded behind obstacles -- retry

    raise RuntimeError(
        f"Could not generate a fully-reachable grid in {max_attempts} attempts; "
        "try a lower obstacle_density."
    )
