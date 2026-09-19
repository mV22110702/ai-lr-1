"""Turn a run into a side-by-side animation: the world, and what the rover thinks.

Two panels, and the gap between them is the whole story of this project. On the
left is the true grid, which the agent never sees. On the right is its belief
map, which starts entirely UNKNOWN and fills in only where the rover has
actually been. Watching the right panel chase the left one is the clearest
picture of partial observability there is -- and when the rover turns for home,
you can see it planning a route across a map with holes still in it.

Writes MP4 by default and GIF on request. The ffmpeg that MP4 needs is not
assumed to be on the system: imageio-ffmpeg ships a binary inside the virtual
environment, and _bundled_ffmpeg() points matplotlib at it, so `uv sync` is the
only setup step. GIF needs nothing beyond pillow and embeds in more places, so
it stays one flag away.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this module only ever writes files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from rover_sim.env import Action
from rover_sim.grid import CellType, Coord

# One colour per CellType, in enum order, so a cell's integer value indexes
# straight into the colour list. UNKNOWN is near-black on purpose: unexplored
# map should read as absence, not as another kind of terrain.
_COLORS: dict[CellType, str] = {
    CellType.FREE: "#e8e4db",
    CellType.OBSTACLE: "#4a4a4a",
    CellType.RESOURCE: "#f2b705",
    CellType.HAZARD: "#d94f3d",
    CellType.BASE: "#2f9e57",
    CellType.UNKNOWN: "#1d232b",
}
_CMAP = ListedColormap([_COLORS[CellType(i)] for i in range(len(CellType))])
# Boundaries at the half-integers, so value 2 lands squarely in the third bin
# and no cell type is ever blended with its neighbour.
_NORM = BoundaryNorm(np.arange(-0.5, len(CellType) + 0.5), _CMAP.N)

_ROVER = "#19c3d6"


@dataclass
class Frame:
    """One moment of a run: the world, the rover's belief, and the rover."""

    step: int
    true_grid: np.ndarray
    belief: np.ndarray | None
    position: Coord
    action: int | None
    energy: int
    samples: int
    total_resources: int
    storm: bool
    mode: str
    reason: str | None = None
    route: list[Coord] | None = None


def _bundled_ffmpeg() -> None:
    """Point matplotlib at the ffmpeg binary that came with imageio-ffmpeg.

    Imported lazily: a GIF-only run should not pay for it, and the failure
    should surface when MP4 is actually asked for.
    """
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "MP4 output needs imageio-ffmpeg; run `uv sync`, or pass --video-format gif"
        ) from exc
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def _style_axes(ax, grid: np.ndarray, title: str) -> None:
    height, width = grid.shape
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.4, alpha=0.2)
    ax.tick_params(
        which="both", bottom=False, left=False, labelbottom=False, labelleft=False
    )
    for spine in ax.spines.values():
        spine.set_edgecolor("#888888")


def save_animation(
    path: Path,
    frames: list[Frame],
    *,
    policy: str,
    seed: int,
    fps: int = 8,
    hold: int = 10,
    fmt: str = "mp4",
) -> Path:
    """Render frames to a video at path, whose suffix is set from fmt.

    hold repeats the last frame at the end. That pause exists because a GIF
    loops: without it the rover appears to teleport from its final cell back
    to base, which reads as a bug rather than a restart.
    """
    if not frames:
        raise ValueError("nothing to animate: no frames were recorded")
    if fmt not in ("mp4", "gif"):
        raise ValueError(f"unsupported video format {fmt!r}: use mp4 or gif")
    path = path.with_suffix(f".{fmt}")

    two_panel = frames[0].belief is not None
    fig, axes = plt.subplots(
        1, 2 if two_panel else 1, figsize=(11.5 if two_panel else 6.2, 4.8)
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    images = []
    trails = []
    rovers = []
    plans = []
    panels: list[tuple[str, str]] = [
        ("true_grid", "The world (the agent never sees this)")
    ]
    if two_panel:
        panels.append(("belief", "What the rover believes"))

    for ax, (attr, title) in zip(axes, panels, strict=True):
        data = getattr(frames[0], attr)
        images.append(ax.imshow(data, cmap=_CMAP, norm=_NORM, interpolation="nearest"))
        (trail,) = ax.plot([], [], "-", color=_ROVER, linewidth=1.2, alpha=0.55)
        # The planned route is drawn only on the belief panel: the point is
        # that the rover plans over the map with the holes in it, not the real
        # one. Dashed, so it reads as intention rather than history.
        (plan,) = ax.plot(
            [],
            [],
            "--",
            color="#ffd166",
            linewidth=1.6,
            alpha=0.95,
            visible=attr == "belief",
        )
        plans.append(plan)
        (rover,) = ax.plot(
            [],
            [],
            "o",
            color=_ROVER,
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.2,
        )
        trails.append(trail)
        rovers.append(rover)
        _style_axes(ax, data, title)

    # UNKNOWN and the planned route exist only on a belief panel, so a
    # single-panel run must not advertise them.
    shown = [c for c in CellType if two_panel or c != CellType.UNKNOWN]
    legend = [
        Patch(facecolor=_COLORS[c], edgecolor="#888888", label=c.name.title())
        for c in shown
    ]
    legend.append(Patch(facecolor=_ROVER, edgecolor="white", label="Rover"))
    if two_panel:
        legend.append(
            Line2D(
                [],
                [],
                color="#ffd166",
                linestyle="--",
                linewidth=1.6,
                label="Planned route",
            )
        )
    # One row fits across the wide two-panel figure; the narrow single-panel
    # one has to wrap, or the labels at the ends get clipped off.
    ncol = len(legend) if two_panel else 4
    bottom = 0.07 if two_panel else 0.14
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=ncol,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    caption = fig.suptitle("", fontsize=11, y=0.97)
    fig.tight_layout(rect=(0, bottom, 1, 0.93))

    # Precomputed so each frame draws the path travelled so far without
    # rescanning the whole run.
    rows = [f.position[0] for f in frames]
    cols = [f.position[1] for f in frames]

    def draw(index: int):
        i = min(index, len(frames) - 1)  # the hold frames repeat the last one
        frame = frames[i]
        for image, (attr, _title) in zip(images, panels, strict=True):
            image.set_data(getattr(frame, attr))
        for trail, rover in zip(trails, rovers, strict=True):
            trail.set_data(cols[: i + 1], rows[: i + 1])
            rover.set_data([cols[i]], [rows[i]])
        # The plan starts from where the rover is standing, so prepend its
        # position -- otherwise the dashes begin one cell ahead and float.
        route = frame.route or []
        for plan in plans:
            plan.set_data(
                [cols[i], *[c for _r, c in route]], [rows[i], *[r for r, _c in route]]
            )
        action = "-" if frame.action is None else Action(frame.action).name
        caption.set_text(
            f"{policy}, seed {seed}   |   step {frame.step}   action {action}   "
            f"energy {frame.energy}   samples {frame.samples}/{frame.total_resources}"
            f"   {frame.mode or '-'}"
            + (f" ({frame.reason})" if frame.reason else "")
            + ("   STORM" if frame.storm else "")
        )
        return [*images, *trails, *rovers, *plans, caption]

    animation = FuncAnimation(
        fig, draw, frames=len(frames) + hold, interval=1000 // max(fps, 1), blit=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "mp4":
        _bundled_ffmpeg()
        # yuv420p is what makes the file playable in browsers and QuickTime
        # rather than only in VLC; it needs even pixel dimensions, which the
        # figure sizes above are chosen to give.
        writer = FFMpegWriter(
            fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]
        )
    else:
        writer = PillowWriter(fps=fps)
    animation.save(path, writer=writer)
    plt.close(fig)
    return path
