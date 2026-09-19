"""Command-line entry point: `uv run python -m rover_sim.cli <command>`."""

from __future__ import annotations

import typer

from rover_sim.grid import generate_grid
from rover_sim.render import render_grid

app = typer.Typer()


# The callback body is empty on purpose: typer collapses a single-command app
# so its command name can be omitted, but more commands land here later (run,
# compare). Registering a callback keeps `show-map` explicit from the start.
@app.callback()
def main() -> None:
    """Exploration rover simulator on a 2D grid."""


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
    try:
        grid, base_pos = generate_grid(
            height=height,
            width=width,
            seed=seed,
            obstacle_density=obstacle_density,
            num_resources=num_resources,
            num_hazards=num_hazards,
        )
    except RuntimeError as exc:
        # Over-dense or over-crowded parameters are user error, not a crash --
        # report them as a CLI message instead of a traceback.
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    render_grid(grid, agent_pos=base_pos)


if __name__ == "__main__":
    app()
