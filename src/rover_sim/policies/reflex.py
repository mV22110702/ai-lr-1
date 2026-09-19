"""The simple reflex agent of the lecture: condition-action rules, no memory."""

from __future__ import annotations

from typing import Any

import numpy as np

from rover_sim.env import MOVES, Action
from rover_sim.grid import CellType


class ReflexPolicy:
    """Acts on the current percept alone and keeps no state between steps.

    This agent is meant to be inadequate, and the shape of its failure is the
    lesson: with no memory it cannot tell "I have already been here" from
    "this is new", so identical percepts produce identical moves and the rover
    paces between the same two cells forever. Milestone 4 exists to fix
    exactly this.
    """

    name = "reflex"

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    def act(self, obs: dict[str, Any]) -> int:
        """The condition-action rule ladder, tried in order."""
        view = obs["local_view"]
        radius = (view.shape[0] - 1) // 2  # derive r from the view, not the env
        here = view[radius, radius]

        # Rule 1: standing on a sample -- take it.
        if here == CellType.RESOURCE:
            return int(Action.COLLECT)

        # Rule 2: a sample is right next door -- step onto it.
        for action, (d_row, d_col) in MOVES.items():
            if view[radius + d_row, radius + d_col] == CellType.RESOURCE:
                return int(action)

        # Rule 3: otherwise walk into whichever direction looks free, always
        # trying them in the same order. The fixed order is what makes the
        # failure visible: nothing is remembered, so the same percept always
        # yields the same move. Choosing randomly among the free directions
        # would wander further, but it would blur the point -- a memoryless
        # agent revisits cells because it cannot know it has seen them.
        for action, (d_row, d_col) in MOVES.items():
            if view[radius + d_row, radius + d_col] != CellType.OBSTACLE:
                return int(action)

        # Rule 4: boxed in on all four sides -- do something arbitrary.
        return int(self._rng.integers(0, len(Action)))
