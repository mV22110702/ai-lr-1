"""Finding a way home across what the rover *believes* the map looks like.

Every function here takes a belief map, never the environment's true grid.
That is the whole point: a planner that could see the real world would make
the partial observability in this project decorative.

Two rules about unknown cells, and the difference matters:

* When **planning a route**, an UNKNOWN cell is impassable. The conservative
  choice: a route home that strolls through cells the rover has never seen
  might walk into a boulder and strand it. It costs nothing here, because the
  rover arrived by walking, so a path of already-seen cells provably exists.
* When **deciding whether exploring is finished**, UNKNOWN is what we are
  looking for, not something to route through.

Breadth-first search is used because every move costs the same number of grid
steps, and BFS is the algorithm that returns the fewest-steps path when that
is true. (Lab 2 revisits BFS properly, next to DFS; lab 3 replaces it with A*.
Here it is only the means to get the rover home.)
"""

from __future__ import annotations

from collections import deque

import numpy as np

from rover_sim.grid import CellType, Coord, neighbors4


def _is_passable(belief_map: np.ndarray, cell: Coord, unknown_passable: bool) -> bool:
    value = belief_map[cell]
    if value == CellType.OBSTACLE:
        return False
    if value == CellType.UNKNOWN:
        return unknown_passable
    return True


def bfs_path(
    belief_map: np.ndarray,
    start: Coord,
    goal: Coord,
    *,
    unknown_passable: bool = False,
) -> list[Coord] | None:
    """Shortest route from start to goal, or None if the belief map has none.

    The returned list excludes start and ends with goal, so its length is the
    number of moves to make and path[0] is the very next cell to step onto.

    How BFS works, in one paragraph: keep a queue of cells to look at, seeded
    with start. Pop the oldest one, and for each of its neighbours that has
    never been queued, record *which cell it was reached from* and push it.
    Because the oldest cell is always popped first, cells come off the queue
    in order of distance from start -- all the one-step cells, then all the
    two-step cells, and so on. So the first time goal is popped, it was
    reached by a shortest route, and walking the "reached from" links back to
    start spells that route out backwards.
    """
    if start == goal:
        return []

    height, width = belief_map.shape
    # came_from does double duty: it is the visited set (a cell in it has been
    # queued already, so it must never be queued twice) and the back-links
    # used to rebuild the path at the end.
    came_from: dict[Coord, Coord] = {start: start}
    queue: deque[Coord] = deque([start])

    while queue:
        current = queue.popleft()
        for neighbour in neighbors4(current, height, width):
            if neighbour in came_from:
                continue
            if not _is_passable(belief_map, neighbour, unknown_passable):
                continue
            came_from[neighbour] = current
            if neighbour == goal:
                return _rebuild(came_from, start, goal)
            queue.append(neighbour)

    return None


def _rebuild(came_from: dict[Coord, Coord], start: Coord, goal: Coord) -> list[Coord]:
    """Walk the back-links from goal to start, then flip them around."""
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path[1:]  # drop start: the rover is already standing there


def reachable_cells(
    belief_map: np.ndarray, start: Coord, *, unknown_passable: bool = False
) -> set[Coord]:
    """Every cell the rover could walk to, according to its belief map.

    The same sweep as bfs_path without a destination -- used to ask "is there
    anything left worth exploring?" rather than "how do I get there?".
    """
    height, width = belief_map.shape
    seen: set[Coord] = {start}
    queue: deque[Coord] = deque([start])
    while queue:
        current = queue.popleft()
        for neighbour in neighbors4(current, height, width):
            if neighbour in seen:
                continue
            if not _is_passable(belief_map, neighbour, unknown_passable):
                continue
            seen.add(neighbour)
            queue.append(neighbour)
    return seen
