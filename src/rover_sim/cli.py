"""Command-line entry point: `uv run python -m rover_sim.cli <command>`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rover_sim.env import Action, RoverEnv
from rover_sim.grid import generate_grid
from rover_sim.logging_utils import Step, write_summary_json, write_trajectory_csv
from rover_sim.policies import ModelBasedPolicy, Policy, RandomPolicy, ReflexPolicy
from rover_sim.render import render_grid

app = typer.Typer()

# Built through small factories rather than stored as bare classes: the
# model-based agent needs the survey dimensions, the other two do not, and a
# uniform (seed, height, width) signature keeps the call site free of
# per-policy branching.
_POLICIES: dict[str, Callable[[int, int, int], Policy]] = {
    "random": lambda seed, _height, _width: RandomPolicy(seed=seed),
    "reflex": lambda seed, _height, _width: ReflexPolicy(seed=seed),
    "model-based": lambda seed, height, width: ModelBasedPolicy(
        seed=seed, height=height, width=width
    ),
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
    policy: str = "model-based",
    seed: int = 42,
    max_steps: int = 200,
    sensor_radius: int = 1,
    max_energy: int = 200,
    storm_prob: float = 0.02,
    trace: int = 30,
    show_map: bool = False,
    show_belief: bool = False,
    log_dir: str = "runs",
) -> None:
    """Run one episode with the chosen policy and print its trajectory.

    --storm-prob and --max-energy are exposed because they are what decides
    *which* of the three "come home now" conditions the rover hits. On the
    defaults a storm is about 87% likely within 100 steps and nearly always
    fires first; --storm-prob 0 lets a run finish its survey instead, and a
    small --max-energy forces the energy-margin case.
    """
    if policy not in _POLICIES:
        known = ", ".join(sorted(_POLICIES))
        typer.secho(
            f"Error: unknown policy '{policy}'. Try: {known}", fg="red", err=True
        )
        raise typer.Exit(code=1)

    console = Console()
    env = RoverEnv(
        max_steps=max_steps,
        sensor_radius=sensor_radius,
        max_energy=max_energy,
        storm_prob=storm_prob,
    )
    agent: Policy = _POLICIES[policy](seed, env.height, env.width)

    agent.reset()
    obs, info = env.reset(seed=seed)

    if show_map:
        console.print(
            f"[bold]True map (seed {seed})[/bold] -- the agent cannot see this:"
        )
        render_grid(env.unwrapped._grid, agent_pos=info["position"], console=console)

    table = Table(title=f"{policy} policy, seed {seed}")
    for column in ("step", "action", "position", "energy", "samples", "storm", "mode"):
        table.add_column(column, justify="right")

    visited: set[tuple[int, int]] = {info["position"]}
    trajectory: list[Step] = []
    total_reward = 0.0
    steps = 0
    mode = ""
    terminated = truncated = False

    for step in range(1, max_steps + 1):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps = step
        visited.add(info["position"])
        mode = str(getattr(agent, "mode", ""))

        trajectory.append(
            Step.build(
                step=step,
                position=info["position"],
                action=action,
                energy=info["energy"],
                samples=info["samples"],
                storm=info["storm"],
                mode=mode,
            )
        )
        if step <= trace:
            table.add_row(
                str(step),
                Action(action).name,
                str(info["position"]),
                str(info["energy"]),
                str(info["samples"]),
                "yes" if info["storm"] else "-",
                mode or "-",
            )
        if terminated or truncated:
            break
        # An agent that has decided it is finished, or that it has failed,
        # ends the episode too. The environment only knows objective facts;
        # giving up is a judgement, and judgements belong to the agent.
        if mode in ("DONE", "FAILED"):
            break

    console.print(table)
    if steps > trace:
        console.print(f"[dim]... {steps - trace} further steps not shown[/dim]")

    reason = getattr(agent, "return_reason", None)
    if terminated and info["mission_complete"]:
        outcome, plain = "[green]mission complete[/green]", "mission_complete"
    elif mode == "DONE":
        outcome, plain = "[green]returned to base[/green]", "returned_to_base"
    elif terminated and info["out_of_energy"]:
        outcome, plain = "[red]out of energy away from base[/red]", "out_of_energy"
    elif mode == "FAILED":
        outcome, plain = "[red]agent reports failure[/red]", "agent_failed"
    elif truncated:
        outcome, plain = "[yellow]truncated (step limit)[/yellow]", "truncated"
    else:
        outcome, plain = "unfinished", "unfinished"
    if reason:
        outcome += f" [dim](went home: {reason})[/dim]"

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

    belief = getattr(agent, "belief_map", None)
    if belief is not None:
        known = int(agent.known_cells)
        total = belief.size
        console.print(
            f"map known to the agent: {known}/{total} cells "
            f"([bold]{known / total:.0%}[/bold])"
        )
        if show_belief:
            console.print("\n[bold]The agent's belief map[/bold] (? = never seen):")
            render_grid(belief, agent_pos=info["position"], console=console)

    if log_dir:
        out = Path(log_dir) / f"{policy}-seed{seed}"
        write_trajectory_csv(out / "trajectory.csv", trajectory)
        write_summary_json(
            out / "summary.json",
            {
                "policy": policy,
                "seed": seed,
                "outcome": plain,
                "return_reason": reason,
                "steps": steps,
                "samples": int(info["samples"]),
                "total_resources": int(info["total_resources"]),
                "energy_left": int(info["energy"]),
                "storm": bool(info["storm"]),
                "base_pos": list(info["base_pos"]),
                "final_pos": list(info["position"]),
                "distinct_cells_visited": len(visited),
                "distinct_cells_per_step": round(reach, 3),
                "map_known_cells": int(agent.known_cells)
                if belief is not None
                else None,
                "map_total_cells": int(belief.size) if belief is not None else None,
                "total_reward": round(total_reward, 2),
            },
        )
        console.print(f"[dim]log written to {out}/[/dim]")


if __name__ == "__main__":
    app()
