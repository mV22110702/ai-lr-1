"""The world the rover acts in: RoverEnv, a gymnasium environment.

This module owns the *true* state of the world -- the real grid, the real
energy level, whether a storm is really happening. An agent never touches any
of it; it only ever sees the observation dict handed back by reset() and
step(). Keeping that wall intact is the point of the assignment, and it is
what lets the agents be tested independently of the world.

Coordinates are (row, col) throughout, including in the observation, matching
numpy indexing. The spec writes position as (x, y); one convention everywhere
was chosen over matching that wording, because the agent indexes its belief
map with exactly what it receives and never has to flip the pair.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rover_sim.grid import CellType, Coord, generate_grid
from rover_sim.render import render_grid


class Action(IntEnum):
    """The rover's actuators -- the assignment's "functions-activators"."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    COLLECT = 4
    WAIT = 5


# (row, col) deltas. Rows grow downward, so UP is -1 along the row axis.
# Public: an agent needs this to line its sensor view up with its actions.
MOVES: dict[Action, Coord] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


class RoverEnv(gym.Env):
    """A partially observable 2D grid world with obstacles, hazards and storms.

    The episode ends on objective facts only, never on the agent's intent:
    success when every resource has been collected and the rover is back at
    base, failure when energy runs out away from base, truncation on the step
    limit. An agent that decides to abort early (storm, low energy) simply
    returns to base; stopping the loop at that point is the run loop's job,
    not the environment's -- the environment must not know agent strategy.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human"],
        "render_fps": 4,
    }

    def __init__(
        self,
        height: int = 12,
        width: int = 16,
        obstacle_density: float = 0.2,
        num_resources: int = 5,
        num_hazards: int = 4,
        sensor_radius: int = 1,
        max_energy: int = 200,
        max_steps: int = 500,
        storm_after_step: int = 30,
        storm_prob: float = 0.02,
        move_cost: int = 1,
        hazard_cost: int = 3,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.height = height
        self.width = width
        self.obstacle_density = obstacle_density
        self.num_resources = num_resources
        self.num_hazards = num_hazards
        self.sensor_radius = sensor_radius
        self.max_energy = max_energy
        self.max_steps = max_steps
        self.storm_after_step = storm_after_step
        self.storm_prob = storm_prob
        self.move_cost = move_cost
        self.hazard_cost = hazard_cost
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(len(Action))

        view = 2 * sensor_radius + 1
        self.observation_space = spaces.Dict(
            {
                "local_view": spaces.Box(
                    low=0, high=int(max(CellType)), shape=(view, view), dtype=np.int8
                ),
                "position": spaces.Box(
                    low=np.zeros(2, dtype=np.int32),
                    high=np.array([height - 1, width - 1], dtype=np.int32),
                    dtype=np.int32,
                ),
                "energy": spaces.Box(
                    low=0, high=max_energy, shape=(1,), dtype=np.int32
                ),
                "samples": spaces.Box(
                    low=0, high=num_resources, shape=(1,), dtype=np.int32
                ),
                "storm": spaces.Discrete(2),
            }
        )

        self._grid_seed: int | None = None
        self._grid = np.zeros((height, width), dtype=np.int8)
        self._base_pos: Coord = (0, 0)
        self._pos: Coord = (0, 0)
        self._energy = max_energy
        self._samples = 0
        self._storm = False
        self._steps = 0
        self._total_resources = 0

    # ------------------------------------------------------------------
    # Perceptors -- the assignment's "functions-perceptors". The observation
    # is assembled from these and nothing else, so what the agent can know is
    # visible in one place.
    # ------------------------------------------------------------------

    def _sense_local(self) -> np.ndarray:
        """Terrain sensor: the (2r+1) square of cells centred on the rover.

        Off-grid cells are reported as OBSTACLE. That is truthful -- the rover
        genuinely cannot move there -- and it spares the agent a special case.
        Reporting UNKNOWN instead would make the map edge look explorable.
        """
        r = self.sensor_radius
        view = np.full((2 * r + 1, 2 * r + 1), CellType.OBSTACLE, dtype=np.int8)
        row, col = self._pos
        for d_row in range(-r, r + 1):
            for d_col in range(-r, r + 1):
                rr, cc = row + d_row, col + d_col
                if 0 <= rr < self.height and 0 <= cc < self.width:
                    view[d_row + r, d_col + r] = self._grid[rr, cc]
        return view

    def _read_position(self) -> Coord:
        """Odometry: the rover always knows where it is."""
        return self._pos

    def _read_energy(self) -> int:
        """Battery gauge."""
        return int(self._energy)

    def _count_samples(self) -> int:
        """Sample-bay counter."""
        return int(self._samples)

    def _sense_weather(self) -> bool:
        """Weather sensor: is a storm running right now?"""
        return bool(self._storm)

    def _observe(self) -> dict[str, Any]:
        return {
            "local_view": self._sense_local(),
            "position": np.array(self._read_position(), dtype=np.int32),
            "energy": np.array([self._read_energy()], dtype=np.int32),
            "samples": np.array([self._count_samples()], dtype=np.int32),
            "storm": int(self._sense_weather()),
        }

    def _get_info(self) -> dict[str, Any]:
        """Ground truth for logs and tests only -- policies must not read this.

        Anything a policy is allowed to know belongs in _observe(); putting it
        here instead would quietly hand the agent x-ray vision.
        """
        return {
            "position": self._pos,
            "base_pos": self._base_pos,
            "energy": int(self._energy),
            "samples": int(self._samples),
            "storm": bool(self._storm),
            "steps": self._steps,
            "at_base": self._pos == self._base_pos,
            "total_resources": self._total_resources,
            "grid_seed": self._grid_seed,
        }

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a fresh episode.

        Passing seed= picks the map and fixes the storm rolls. A reset without
        a seed reuses the previous map rather than drawing a new one: runs in
        this project are meant to be repeatable by default, and the caller can
        always ask for a different world explicitly.
        """
        super().reset(seed=seed)

        if seed is not None:
            self._grid_seed = seed  # 1. caller named a seed -> use it
        elif self._grid_seed is None:  # 2. first ever reset, no seed -> invent one
            self._grid_seed = int(self.np_random.integers(0, 2**31 - 1))
            # 3. (implicit) otherwise keep the old seed

        # Existing grid may be modified from prev round (a resource is collected) -> Create a new one
        self._grid, self._base_pos = generate_grid(
            height=self.height,
            width=self.width,
            seed=self._grid_seed,
            obstacle_density=self.obstacle_density,
            num_resources=self.num_resources,
            num_hazards=self.num_hazards,
        )
        self._total_resources = int((self._grid == CellType.RESOURCE).sum())

        self._pos = self._base_pos
        self._energy = self.max_energy
        self._samples = 0
        self._storm = False
        self._steps = 0

        return self._observe(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one action and advance the world by a step."""
        act = Action(int(action))
        self._steps += 1
        self._maybe_start_storm()

        collected = False
        entered_hazard = False

        if act in MOVES:
            d_row, d_col = MOVES[act]
            target = (self._pos[0] + d_row, self._pos[1] + d_col)
            if self._is_passable(target):
                self._pos = target
                entered_hazard = self._grid[target] == CellType.HAZARD
                cost = self.hazard_cost if entered_hazard else self.move_cost
            else:
                # Bumping into a wall is a legal action that simply fails. It
                # still costs energy, which is both realistic and gives a
                # learning agent something to react to.
                cost = self.move_cost
        elif act is Action.COLLECT:
            cost = self.move_cost
            if self._grid[self._pos] == CellType.RESOURCE:
                self._grid[self._pos] = CellType.FREE
                self._samples += 1
                collected = True
        else:  # Action.WAIT
            cost = self.move_cost

        if self._storm:
            cost *= 2
        self._energy = max(0, self._energy - cost)

        at_base = self._pos == self._base_pos
        if at_base:
            self._energy = self.max_energy  # recharge coupling, automatic

        mission_complete = at_base and self._samples == self._total_resources
        out_of_energy = self._energy <= 0
        terminated = bool(mission_complete or out_of_energy)
        truncated = bool(self._steps >= self.max_steps and not terminated)

        reward = self._compute_reward(
            collected=collected,
            entered_hazard=entered_hazard,
            mission_complete=mission_complete,
            out_of_energy=out_of_energy,
        )

        info = self._get_info()
        info["mission_complete"] = mission_complete
        info["out_of_energy"] = out_of_energy
        return self._observe(), reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode == "human":
            render_grid(self._grid, agent_pos=self._pos)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        *,
        collected: bool,
        entered_hazard: bool,
        mission_complete: bool,
        out_of_energy: bool,
    ) -> float:
        """All reward shaping lives here, so it is easy to discuss and tweak.

        Only the optional PPO agent of Milestone 10 consumes this; the
        hand-written policies ignore reward entirely.
        """
        reward = -0.1  # per-step cost: dithering is never free
        if collected:
            reward += 10.0
        if entered_hazard:
            reward -= 0.5
        if mission_complete and self._samples > 0:
            reward += 20.0
        if out_of_energy:
            reward -= 20.0
        return reward

    def _maybe_start_storm(self) -> None:
        """Storms can start only after storm_after_step, and never stop.

        Persistent rather than flickering: the storm is the "unfavourable
        conditions" trigger for aborting a mission, and one that switched off
        again would let an agent dither instead of committing to go home.
        """
        if (
            not self._storm
            and self._steps > self.storm_after_step
            and self.np_random.random() < self.storm_prob
        ):
            self._storm = True

    def _is_passable(self, pos: Coord) -> bool:
        row, col = pos
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        return bool(self._grid[pos] != CellType.OBSTACLE)
