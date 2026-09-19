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

On top of the belief map sits the mode machine the lab asks for
(EXPLORING -> RETURNING -> DONE, with FAILED for a flat battery). It is what
turns "wander and grab things" into "explore until the goal is met or
conditions turn unfavourable, then come home".
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from rover_sim.env import MOVES, Action
from rover_sim.grid import CellType, Coord, neighbors4
from rover_sim.logging_utils import Step
from rover_sim.pathfinding import bfs_path, reachable_cells

Mode = Literal["EXPLORING", "RETURNING", "DONE", "FAILED"]

# The inverse of MOVES: given the cell I want to step onto, which action gets
# me there? Needed to turn a planned path back into actions.
_DELTA_TO_ACTION: dict[Coord, Action] = {delta: act for act, delta in MOVES.items()}


class ModelBasedPolicy:
    """Keeps a belief map and a visit history, and explores using both.

    Grid *dimensions* are handed to the constructor rather than perceived.
    That is not a hole in the agent/environment wall: the size of the survey
    area is mission briefing, known before the rover lands. Its *contents* are
    exactly what the agent has to discover. The same goes for the energy
    tariff -- a rover knows what its own motors cost to run.
    """

    name = "model-based"

    def __init__(
        self,
        seed: int | None = None,
        height: int = 12,
        width: int = 16,
        move_cost: int = 1,
        hazard_cost: int = 3,
        safety_margin: float = 1.3,
    ) -> None:
        self.height = height
        self.width = width
        self.move_cost = move_cost
        self.hazard_cost = hazard_cost
        # Head home while the trip still costs 30% less than the battery
        # holds. The margin is not decoration: the route is planned over a
        # belief map that can be wrong, a storm can double every cost
        # mid-journey, and arriving with an empty battery one cell short of
        # base scores exactly the same as never leaving.
        self.safety_margin = safety_margin
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
        self.return_reason: str | None = None
        self.trajectory: list[Step] = []
        self._steps = 0
        self._route_home: list[Coord] = []
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
            # of the environment. Worth remembering: there is no percept for
            # "where is home", and the route back has to aim somewhere.
            self.base_pos = new_pos

        self._update_mode()

    # ------------------------------------------------------------------
    # The mode machine: explore, then come home
    # ------------------------------------------------------------------

    def _update_mode(self) -> None:
        """EXPLORING -> RETURNING -> DONE, with FAILED if the battery dies.

        Re-evaluated from scratch every single step, never latched on a
        one-off reading. Under partial observability the numbers that decide
        this keep changing: the route home gets shorter as the belief map
        improves, and a storm doubles its cost the moment it starts.
        """
        if self.mode in ("DONE", "FAILED"):
            return

        if self.energy <= 0:
            self.mode = "FAILED"
            return

        # Plan the way home first: two of the three triggers below need to
        # know what the trip costs, and RETURNING needs the route anyway.
        self._route_home = self._plan_route_home()

        if self.mode == "EXPLORING":
            reason = self._reason_to_go_home()
            if reason is not None:
                self.mode = "RETURNING"
                self.return_reason = reason

        if self.mode == "RETURNING" and self.position == self.base_pos:
            self.mode = "DONE"

    def _reason_to_go_home(self) -> str | None:
        """The three conditions from the brief, in priority order.

        Cheapest and most certain first: a finished survey is a fact about the
        belief map, a storm is a direct percept, and the energy margin is an
        estimate built on top of a plan.
        """
        if self._survey_complete():
            return "survey complete"
        if self.storm:
            return "storm"
        if self._route_home and self.energy <= self._trip_cost() * self.safety_margin:
            return "energy margin"
        return None

    def _survey_complete(self) -> bool:
        """Nothing left worth walking to: no reachable resource, no frontier.

        "Reachable" is judged over known cells only, so a pocket of the map
        sealed off behind obstacles does not keep the rover out forever. A
        frontier cell is a known, walkable cell that touches something
        unknown -- somewhere the rover could stand and learn something new.
        """
        reach = reachable_cells(self.belief_map, self.position)
        if any(self.belief_map[cell] == CellType.RESOURCE for cell in reach):
            return False
        for cell in reach:
            for neighbour in neighbors4(cell, self.height, self.width):
                if self.belief_map[neighbour] == CellType.UNKNOWN:
                    return False
        return True

    def _plan_route_home(self) -> list[Coord]:
        """The cells to step on to reach base, shortest first. Empty if home.

        Unknown cells are impassable, which sounds risky and is not: the rover
        walked out here, so a route made only of cells it has already seen
        provably exists. The fallback that lets the planner gamble on unknown
        cells is there for the pathological case where the belief map got
        stale, and should essentially never fire.
        """
        if self.base_pos is None or self.position == self.base_pos:
            return []
        path = bfs_path(self.belief_map, self.position, self.base_pos)
        if path is None:
            path = bfs_path(
                self.belief_map, self.position, self.base_pos, unknown_passable=True
            )
        return path or []

    def _trip_cost(self) -> int:
        """Energy the planned route home will burn, on current information."""
        cost = 0
        for cell in self._route_home:
            cost += (
                self.hazard_cost
                if self.belief_map[cell] == CellType.HAZARD
                else self.move_cost
            )
        return cost * 2 if self.storm else cost

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def act(self, obs: dict[str, Any]) -> int:
        """Perceive, update the internal state, then apply the rules."""
        self.update_state(obs)
        action = self._choose_action()

        self._steps += 1
        self.trajectory.append(
            Step.build(
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
        if self.mode in ("DONE", "FAILED"):
            return int(Action.WAIT)

        # Worth one step in either mode: the rover is already standing on it,
        # and a sample left behind is the whole point of the mission missed.
        if self.belief_map[self.position] == CellType.RESOURCE:
            return int(Action.COLLECT)

        if self.mode == "RETURNING":
            return self._step_along_route()

        return self._explore()

    def _step_along_route(self) -> int:
        """Take the first move of the planned route home."""
        if not self._route_home:
            return int(Action.WAIT)  # no way back on current knowledge
        next_cell = self._route_home[0]
        delta = (next_cell[0] - self.position[0], next_cell[1] - self.position[1])
        return int(_DELTA_TO_ACTION[delta])

    def _explore(self) -> int:
        """Greedy exploration: step onto the most promising neighbour."""
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
           prefer the one facing more of the unmapped map.
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
    def route_home(self) -> list[Coord]:
        """The currently planned way back, for logging and visualisation.

        A copy: handing out the live list would let a caller edit the rover's
        plan out from under it.
        """
        return list(self._route_home)

    @property
    def known_cells(self) -> int:
        """How much of the map the rover has mapped."""
        return int((self.belief_map != CellType.UNKNOWN).sum())
