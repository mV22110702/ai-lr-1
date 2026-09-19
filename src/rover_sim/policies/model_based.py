"""The model-based agent: the first one that remembers anything.

The reflex agent of Milestone 3 fails for one reason only -- it has no memory,
so two visits to the same cell look identical to it and it makes the same move
both times. This agent fixes that by keeping an *internal state*: a private
picture of the world that it carries from step to step and updates from every
percept. Slides 16-17 of the lecture call this the model-based reflex agent,
and its loop is:

    state = UPDATE_STATE(state, last_action, percept)
    rule  = RULE_MATCH(state, rules)
    action = rule.action

Two things make the internal state a *model* rather than a log:

* the **transition model** -- "if I did X, the world should now look like Y".
  Here: I asked to move UP, so I expect to be one row higher.
* the **sensor model** -- "this percept means the world is like Y".
  Here: this 3x3 view means these nine cells hold these terrain types.

`update_state()` merges both into one belief map. The merge is where the
inference happens: when the prediction and the observation disagree (I asked
to move UP and I did *not* move), the agent learns something it never directly
saw -- the cell above it is blocked.

The belief map is emphatically *not* the true grid. It starts entirely
UNKNOWN, fills in only where the rover has actually been, and may be stale
where the world changed out of sight. Everything that plans a route -- getting
home at the end of this lab, the search algorithms of labs 2 and 3 -- plans
over this map, never over the environment's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from rover_sim.env import MOVES, Action
from rover_sim.grid import CellType, Coord, neighbors4

Mode = Literal["EXPLORING", "RETURNING", "DONE", "FAILED"]


@dataclass(frozen=True)
class Step:
    """One row of the agent's own trajectory log.

    The agent records what *it* knew at the time, not the truth: this is the
    evidence for the report in Milestone 7, and it has to be honest about
    partial observability. Frozen because a logged step is history -- nothing
    should be able to edit it after the fact.
    """

    step: int
    position: Coord
    action: int
    energy: int
    samples: int
    storm: bool
    mode: Mode


class ModelBasedPolicy:
    """Keeps a belief map and a visit history, and explores using both.

    Grid *dimensions* are handed to the constructor rather than perceived.
    That is not a hole in the agent/environment wall: the size of the survey
    area is mission briefing, known before the rover lands. Its *contents* are
    exactly what the agent has to discover.
    """

    name = "model-based"

    def __init__(
        self, seed: int | None = None, height: int = 12, width: int = 16
    ) -> None:
        self.height = height
        self.width = width
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Wipe the internal state -- a fresh rover on a fresh, unknown map."""
        self.belief_map: np.ndarray = np.full(
            (self.height, self.width), CellType.UNKNOWN, dtype=np.int8
        )
        self.visited: set[Coord] = set()
        self.last_action: int | None = None  # None, not a sentinel int: before
        # the first step there genuinely is no last action to reason about
        self.samples = 0
        self.energy = 0
        self.storm = False
        self.position: Coord = (0, 0)
        self.base_pos: Coord | None = None
        self.mode: Mode = "EXPLORING"
        self.trajectory: list[Step] = []
        self._steps = 0
        # visited answers "have I been here"; the counts answer "how often",
        # which is what breaks a tie between two cells the rover has both
        # seen. Keeping the plain set as well is deliberate -- it is the
        # attribute the rest of the project and the report talk about.
        self._visit_counts: dict[Coord, int] = {}
        self._rng = np.random.default_rng(self._seed)

    def update_state(self, obs: dict[str, Any]) -> None:
        """UPDATE_STATE from the lecture: prediction + observation -> belief.

        Called once per percept, before any decision is made. Order matters:
        the transition model is checked *against* the new odometry reading, so
        it has to run before self.position is overwritten.
        """
        new_pos: Coord = (int(obs["position"][0]), int(obs["position"][1]))

        # --- transition model -------------------------------------------
        # What should have happened, given what I last did? If I asked to move
        # and my position did not change, something solid is in the way. This
        # is knowledge derived from acting, not from sensing -- exactly the
        # thing a memoryless agent can never have.
        if self.last_action is not None and self.last_action in set(MOVES):
            d_row, d_col = MOVES[Action(self.last_action)]
            expected = (self.position[0] + d_row, self.position[1] + d_col)
            if new_pos != expected and self._in_bounds(expected):
                self.belief_map[expected] = CellType.OBSTACLE

        # --- sensor model ------------------------------------------------
        # The local view is ground truth about a small square. Copy it into
        # the belief map, translating view coordinates into map coordinates.
        view = obs["local_view"]
        radius = (view.shape[0] - 1) // 2
        for d_row in range(-radius, radius + 1):
            for d_col in range(-radius, radius + 1):
                cell = (new_pos[0] + d_row, new_pos[1] + d_col)
                if self._in_bounds(cell):
                    self.belief_map[cell] = view[d_row + radius, d_col + radius]

        # --- bookkeeping ---------------------------------------------------
        self.position = new_pos
        self.visited.add(new_pos)
        self._visit_counts[new_pos] = self._visit_counts.get(new_pos, 0) + 1
        self.energy = int(obs["energy"][0])
        self.samples = int(obs["samples"][0])
        self.storm = bool(obs["storm"])
        if self.base_pos is None:
            # The first cell the rover ever stands on is base, by construction
            # of the environment. Worth remembering: Milestone 5 has to plan a
            # route back to it.
            self.base_pos = new_pos

        self._update_mode()

    def _update_mode(self) -> None:
        """The mode state machine -- only the transitions this agent can honour.

        EXPLORING -> FAILED is the one that is real today. The other three
        (going home when the goal is met, when energy runs low, when a storm
        starts) all need a route home to be worth anything, and the route
        planner is the next step. Declaring RETURNING here while still
        wandering greedily would be a lie told by the state machine.
        """
        if self.energy <= 0:
            self.mode = "FAILED"

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def act(self, obs: dict[str, Any]) -> int:
        """Perceive, update the internal state, then apply the rules."""
        self.update_state(obs)
        action = self._choose_action()

        self._steps += 1
        self.trajectory.append(
            Step(
                step=self._steps,
                position=self.position,
                action=action,
                energy=self.energy,
                samples=self.samples,
                storm=self.storm,
                mode=self.mode,
            )
        )
        self.last_action = action
        return action

    def _choose_action(self) -> int:
        """RULE_MATCH: the rules now read the belief map, not just the percept."""
        if self.belief_map[self.position] == CellType.RESOURCE:
            return int(Action.COLLECT)

        candidates = [
            (action, cell)
            for action, cell in self._neighbour_moves()
            if self.belief_map[cell] != CellType.OBSTACLE
        ]
        if not candidates:
            return int(Action.WAIT)  # walled in; nothing sensible to do

        for action, cell in candidates:
            if self.belief_map[cell] == CellType.RESOURCE:
                return int(action)

        # Shuffle first so that equally-good cells are picked between at
        # random: a fixed order is what let the reflex agent fall into a
        # two-cell loop, and ties happen constantly on an open map.
        order = self._rng.permutation(len(candidates))
        best = min((candidates[i] for i in order), key=lambda ac: self._cost_of(ac[1]))
        return int(best[0])

    def _cost_of(self, cell: Coord) -> tuple[int, int, int]:
        """Rank a neighbouring cell -- lower is better, compared left to right.

        1. how often I have stood there: never-visited cells beat visited ones,
           and among visited ones the stalest wins. This single number is what
           makes re-exploration impossible to fall into: every revisit makes
           the cell less attractive than its neighbours.
        2. whether it is a hazard: hazards cost 3 energy instead of 1, so
           between two equally fresh cells, take the safe one.
        3. how much unknown territory it touches, negated: of two fresh cells,
           prefer the one facing more of the unmapped map. This is a cheap
           stand-in for the real frontier search of Milestone 6.
        """
        visits = self._visit_counts.get(cell, 0)
        is_hazard = int(self.belief_map[cell] == CellType.HAZARD)
        unknown_around = sum(
            self.belief_map[n] == CellType.UNKNOWN
            for n in neighbors4(cell, self.height, self.width)
        )
        return (visits, is_hazard, -unknown_around)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _neighbour_moves(self) -> list[tuple[Action, Coord]]:
        """The four moves and the cells they lead to, minus those off the map."""
        moves = []
        for action, (d_row, d_col) in MOVES.items():
            cell = (self.position[0] + d_row, self.position[1] + d_col)
            if self._in_bounds(cell):
                moves.append((action, cell))
        return moves

    def _in_bounds(self, cell: Coord) -> bool:
        row, col = cell
        return 0 <= row < self.height and 0 <= col < self.width

    @property
    def known_cells(self) -> int:
        """How much of the map the rover has mapped -- the Milestone 4 metric."""
        return int((self.belief_map != CellType.UNKNOWN).sum())
