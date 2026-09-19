"""Command-line entry point: `uv run python -m rover_sim.cli <command>`."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from rover_sim.env import Action, RoverEnv
from rover_sim.grid import generate_grid
from rover_sim.policies import Policy, RandomPolicy, ReflexPolicy
from rover_sim.render import render_grid

app = typer.Typer()

_POLICIES: dict[str, type] = {
    "random": RandomPolicy,
    "reflex": ReflexPolicy,
}


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


@app.command(name="run")
def run(
    policy: str = "reflex",
    seed: int = 42,
    max_steps: int = 200,
    sensor_radius: int = 1,
    trace: int = 30,
    show_map: bool = False,
) -> None:
    """Run one episode with the chosen policy and print its trajectory."""
    if policy not in _POLICIES:
        known = ", ".join(sorted(_POLICIES))
        typer.secho(
            f"Error: unknown policy '{policy}'. Try: {known}", fg="red", err=True
        )
        raise typer.Exit(code=1)

    console = Console()
    env = RoverEnv(max_steps=max_steps, sensor_radius=sensor_radius)
    agent: Policy = _POLICIES[policy](seed=seed)

    agent.reset()
    obs, info = env.reset(seed=seed)

    if show_map:
        console.print(
            f"[bold]True map (seed {seed})[/bold] -- the agent cannot see this:"
        )
        render_grid(env.unwrapped._grid, agent_pos=info["position"], console=console)

    table = Table(title=f"{policy} policy, seed {seed}")
    for column in ("step", "action", "position", "energy", "samples", "storm"):
        table.add_column(column, justify="right")

    visited: set[tuple[int, int]] = {info["position"]}
    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    for step in range(1, max_steps + 1):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps = step
        visited.add(info["position"])

        if step <= trace:
            table.add_row(
                str(step),
                Action(action).name,
                str(info["position"]),
                str(info["energy"]),
                str(info["samples"]),
                "yes" if info["storm"] else "-",
            )
        if terminated or truncated:
            break

    console.print(table)
    if steps > trace:
        console.print(f"[dim]... {steps - trace} further steps not shown[/dim]")

    if terminated and info["mission_complete"]:
        outcome = "[green]mission complete[/green]"
    elif terminated and info["out_of_energy"]:
        outcome = "[red]out of energy away from base[/red]"
    elif truncated:
        outcome = "[yellow]truncated (step limit)[/yellow]"
    else:
        outcome = "unfinished"

    # Distinct cells per step is the number that exposes a memoryless agent:
    # an explorer approaches 1.0, a rover pacing between two cells approaches 0.
    reach = len(visited) / steps if steps else 0.0
    console.print(
        f"outcome: {outcome}\n"
        f"steps: {steps}   samples: {info['samples']}/{info['total_resources']}   "
        f"energy left: {info['energy']}\n"
        f"distinct cells visited: {len(visited)} "
        f"([bold]{reach:.2f}[/bold] per step)   total reward: {total_reward:.1f}"
    )


if __name__ == "__main__":
    app()
