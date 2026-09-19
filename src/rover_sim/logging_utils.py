"""Writing down what happened: the trajectory log (lab 1, bullet 5).

Two files per run, because they answer different questions:

* `trajectory.csv` -- one row per step, the whole run. Opens in any
  spreadsheet, which is what makes it evidence rather than scrollback.
* `summary.json` -- the one-line verdict for the run: how it ended, how long
  it took, how much got collected and mapped.

A deliberate distinction worth knowing: this module records what the
*experimenter* saw. The model-based agent separately keeps its own
`trajectory` list, which is part of its internal state -- what the rover
itself remembers. They happen to hold the same rows here; they are not the
same thing, and only the agent's copy is allowed to influence its decisions.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rover_sim.env import Action
from rover_sim.grid import Coord


@dataclass(frozen=True)
class Step:
    """One row of a trajectory.

    Frozen: a logged step is history, and nothing should be able to edit it
    after the fact.
    """

    step: int
    row: int
    col: int
    action: int
    energy: int
    samples: int
    storm: bool
    mode: str

    @classmethod
    def build(
        cls,
        *,
        step: int,
        position: Coord,
        action: int,
        energy: int,
        samples: int,
        storm: bool,
        mode: str,
    ) -> Step:
        """Convenience constructor that splits (row, col) into two columns.

        A tuple in a CSV cell would be written as the string "(4, 7)", which
        no spreadsheet can sort or plot. Two integer columns can.
        """
        return cls(
            step=step,
            row=int(position[0]),
            col=int(position[1]),
            action=int(action),
            energy=int(energy),
            samples=int(samples),
            storm=bool(storm),
            mode=mode,
        )

    @property
    def position(self) -> Coord:
        return (self.row, self.col)


def write_trajectory_csv(path: Path, steps: list[Step]) -> None:
    """Write one row per step, with a readable action name alongside its code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "row",
        "col",
        "action",
        "action_name",
        "energy",
        "samples",
        "storm",
        "mode",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in steps:
            row = asdict(step)
            row["action_name"] = Action(step.action).name
            writer.writerow(row)


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    """Write the run verdict. indent=2 so a human can read it without tooling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
