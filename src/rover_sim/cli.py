"""Command-line entry point: `uv run python -m rover_sim.cli <command>`."""

from __future__ import annotations

import typer

from rover_sim.grid import generate_grid
from rover_sim.render import render_grid

app = typer.Typer()


@app.callback()
def main() -> None:
    """Rover exploration simulator.

    An empty callback, on purpose: typer collapses a single-command app so
    the command name can be omitted, but this project always adds more
    commands later (run, compare) -- registering a callback keeps
    `show-map` required and explicit from the start.
    """


@app.command(name="show-map")
def show_map(
    seed: int = 42,
    height: int = 12,
    width: int = 16,
    obstacle_density: float = 0.2,
    num_resources: int = 5,
    num_hazards: int = 4,
) -> None:
    """Generate a grid with the given seed and print it to the terminal."""
    grid, base_pos = generate_grid(
        height=height,
        width=width,
        seed=seed,
        obstacle_density=obstacle_density,
        num_resources=num_resources,
        num_hazards=num_hazards,
    )
    render_grid(grid, agent_pos=base_pos)


if __name__ == "__main__":
    app()