"""The baseline agent: act at random, remember nothing."""

from __future__ import annotations

from typing import Any

import numpy as np

from rover_sim.env import Action


class RandomPolicy:
    """Picks a uniformly random action every step.

    The control that every other agent is measured against in Milestone 9: an
    agent that cannot beat random is not doing anything useful. It carries its
    own generator instead of borrowing the environment's, so that the agent's
    choices and the world's weather stay independently reproducible.
    """

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        # Rebuild the generator so replaying an episode replays the same
        # choices; a policy that kept drawing from a spent stream would make
        # "same seed, same run" false.
        self._rng = np.random.default_rng(self._seed)

    def act(self, obs: dict[str, Any]) -> int:
        return int(self._rng.integers(0, len(Action)))
