"""Turn a grid into readable, colored terminal output.

Kept separate from grid.py on purpose: world generation has zero
knowledge of how (or whether) it gets displayed. render.py only reads a
grid, it never creates or mutates one.
"""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.text import Text

from rover_sim.grid import CellType, Coord

# One (character, color-style) pair per cell type. rich understands plain
# color names like "red" as well as combinations like "bold white on grey23"
# (bold white text on a dark grey background).
_STYLE: dict[CellType, tuple[str, str]] = {
    CellType.FREE: (".", "grey50"),
    CellType.OBSTACLE: ("#", "bold white on grey23"),
    CellType.RESOURCE: ("*", "bold yellow"),
    CellType.HAZARD: ("!", "bold red"),
    CellType.BASE: ("B", "bold green"),
    CellType.UNKNOWN: ("?", "grey35"),
}

_AGENT_CHAR = "R"
_AGENT_STYLE = "bold cyan"


def render_grid(
    grid: np.ndarray,
    agent_pos: Coord | None = None,
    console: Console | None = None,
) -> None:
    """Print grid to the terminal, one colored character per cell.

    agent_pos, if given, is drawn as 'R' on top of whatever cell type is
    underneath it -- the true cell isn't overwritten, only its on-screen
    character is, for this one print call.
    """
    console = console if console is not None else Console()
    height, width = grid.shape

    text = Text()
    for row in range(height):
        for col in range(width):
            if agent_pos is not None and (row, col) == tuple(agent_pos):
                char, style = _AGENT_CHAR, _AGENT_STYLE
            else:
                char, style = _STYLE[CellType(grid[row, col])]
            text.append(char + " ", style=style)
        text.append("\n")
    console.print(text)
