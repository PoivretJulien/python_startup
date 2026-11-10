"""Example 7 — a gauge: a gradient rectangle and a marker moved by the keyboard.

A tall rounded rectangle is filled with a five-colour gradient that runs from
the bottom to the top: blue, yellow, green, orange, red. A tiny triangle,
pointing up, is drawn on top of it and marks the level. The Up and Down arrow
keys move it; hold a key and it keeps moving.

From the project root (after running the installer in toolchain/):

    .venv/bin/python examples/07_gradient_gauge/main.py            # reactive (default)
    .venv/bin/python examples/07_gradient_gauge/main.py --poll     # always redraw

A gradient is two lists of the same length::

    STOPS  = [0.0,  0.25,   0.5,   0.75,   1.0]     # where, from 0.0 to 1.0
    COLORS = [BLUE, YELLOW, GREEN, ORANGE, RED]     # which colour is there

and a direction: ``(0, -1)`` points up on the screen, so position 0.0 (blue) is
at the bottom of the rectangle and 1.0 (red) at the top. Between two stops the
graphics card interpolates the colour, pixel by pixel.

``level`` is the only thing the program remembers: a float from 0.0 to 1.0.
The marker's height, the percentage and the colour of the round lamp are all
computed from it on every frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The drawing library is the graphics/ folder of the project.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "graphics"))

import app3d  # noqa: E402
import screen2d  # noqa: E402

BLUE = (0.15, 0.35, 1.00, 1.0)  # r, g, b, a
YELLOW = (1.00, 0.90, 0.10, 1.0)
GREEN = (0.20, 0.85, 0.30, 1.0)
ORANGE = (1.00, 0.55, 0.10, 1.0)
RED = (1.00, 0.15, 0.15, 1.0)

# The gradient: one colour per stop, the two lists side by side.
STOPS = [0.0, 0.25, 0.5, 0.75, 1.0]
COLORS = [BLUE, YELLOW, GREEN, ORANGE, RED]
UP = (0, -1)  # screen y grows down, so "up" is minus y

WHITE = (1.00, 1.00, 1.00, 1.0)
DARK = (0.06, 0.06, 0.07, 1.0)
LABEL = (0.85, 0.92, 1.00)
MUTED = (0.55, 0.55, 0.62)

GAUGE_ANCHOR = (90, 50)  # top-left corner of the gauge
GAUGE_SIZE = (90, 400)  # width, height
GAUGE_RADIUS = 26
TRAVEL = 30  # the marker stays this far from the two ends
STEP = 0.02  # how much one key press changes the level


def main() -> int:
    app = app3d.App("example 7 — gradient gauge", size=(560, 500))
    ui = app.aa

    level = 0.5  # the whole state: 0.0 = bottom (blue), 1.0 = top (red)

    def draw_frame() -> None:
        nonlocal level

        # --- 1. the keyboard moves the level --------------------------------
        # repeat=True: a held arrow keeps "pressing", as in a text field.
        if app.key_pressed("UP", repeat=True):
            level += STEP
        if app.key_pressed("DOWN", repeat=True):
            level -= STEP
        level = min(max(level, 0.0), 1.0)  # stay on the gauge

        # --- 2. the gauge: one gradient rounded rectangle -------------------
        ui.draw_screen_gradient_rect(GAUGE_ANCHOR, GAUGE_SIZE, STOPS, COLORS, UP, GAUGE_RADIUS)

        # --- 3. the marker: a tiny triangle pointing up, on top of it ----------
        left, top = GAUGE_ANCHOR
        width, height = GAUGE_SIZE
        bottom = top + height
        # level 0 -> near the bottom, level 1 -> near the top.
        tip_y = (bottom - TRAVEL) - level * (height - 2 * TRAVEL)
        centre_x = left + width / 2
        # The triangle is placed by the middle of its base; its tip is `height`
        # further along its direction. A slightly larger dark one behind it
        # keeps it readable on every colour.
        ui.draw_screen_triangle((centre_x, tip_y + 19), 30, 23, UP, DARK)
        ui.draw_screen_triangle((centre_x, tip_y + 16), 22, 16, UP, WHITE)

        # --- 4. read-outs, all computed from `level` ---------------------------
        # The colour of the gradient at the marker: the same interpolation the
        # graphics card does, asked for one position.
        here = screen2d.gradient_color(STOPS, COLORS, level)
        ui.draw_screen_line((left + width + 14, tip_y), (290, tip_y), here, 2)
        ui.draw_screen_circle((330, tip_y), 26, here)
        ui.draw_screen_text(370, tip_y + 7, f"{level * 100:.0f} %", 18.0, LABEL)

        # The same gradient on a disc: same two lists, another method.
        ui.draw_screen_gradient_circle((495, 250), 34, STOPS, COLORS, UP)

        ui.draw_screen_text(20, 480, "Up / Down arrows move the marker   Esc quits", 12.0, MUTED)

    app.run(draw_frame)
    return 0


if __name__ == "__main__":
    sys.exit(main())
