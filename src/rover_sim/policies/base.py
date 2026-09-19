"""What every agent must look like: the Policy protocol.

A policy is the lecture's *agent function* -- percept in, action out. Nothing
here knows about grids, energy or storms: a policy only ever sees the
observation dict the environment hands it. That is what keeps agents testable
on their own and stops them reading the true state of the world.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Policy(Protocol):
    """The agent function: observation -> action.

    A Protocol rather than a base class: any object with these two methods
    counts as a Policy, no inheritance required. The alternative, a shared
    abstract parent, would force four unrelated agents to agree on a common
    ancestor for no benefit -- they share an interface, not an implementation.
    """

    def reset(self) -> None:
        """Forget everything; called once before each episode begins."""
        ...

    def act(self, obs: dict[str, Any]) -> int:
        """Choose one action from one percept."""
        ...
