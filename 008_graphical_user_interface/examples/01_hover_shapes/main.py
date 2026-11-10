"""Example 1 — two screen-space shapes that react to the mouse.

A rounded rectangle and a circle, both drawn in screen space (window pixels,
top-left origin). Each one is purple, and turns yellow while the cursor is over
it. A sharp rectangle (radius 0) sits below them to show the fast path.

From the project root (after running the installer in toolchain/):

    .venv/bin/python examples/01_hover_shapes/main.py            # reactive (default)
    .venv/bin/python examples/01_hover_shapes/main.py --poll     # always redraw

The whole program is ``draw_frame()``: describe a shape once, ask whether the
cursor is in it, draw it with the colour that answer picks. The hit test and
the draw call take the same description, in the same order.

Reactive by default: the window redraws only when the mouse moves or a button
changes, and costs nothing while idle. Esc or the close button quits.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The drawing library is the graphics/ folder of the project.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "graphics"))

import app3d  # noqa: E402
from vec2 import Vec2  # noqa: E402

PURPLE = (0.58, 0.25, 0.90, 1.0)  # r, g, b, a
YELLOW = (1.00, 0.85, 0.10, 1.0)
LABEL = (0.85, 0.92, 1.00)

# --- the two shapes, described once ---------------------------------------
RECT_ANCHOR = Vec2(60, 90)  # top-left corner, window pixels (a tuple works too)
RECT_SIZE = (260, 150)  # width, height
RECT_RADIUS = 28  # the four rounded corners

CIRCLE_CENTER = (500, 165)
CIRCLE_RADIUS = 75

SHARP_ANCHOR = (60, 290)
SHARP_SIZE = (515, 44)


def main() -> int:
    app = app3d.App("example 1 — hover a shape", size=(640, 400))
    ui = app.aa  # the renderer: every draw_* and in_* method lives on it

    def draw_frame() -> None:
        cursor = app.cursor()  # (x, y) in window pixels

        over_rect = ui.in_screen_rect(cursor, RECT_ANCHOR, RECT_SIZE, RECT_RADIUS)
        ui.draw_screen_rect(
            RECT_ANCHOR, RECT_SIZE, YELLOW if over_rect else PURPLE, RECT_RADIUS
        )

        over_circle = ui.in_screen_circle(cursor, CIRCLE_CENTER, CIRCLE_RADIUS)
        ui.draw_screen_circle(
            CIRCLE_CENTER, CIRCLE_RADIUS, YELLOW if over_circle else PURPLE
        )

        # radius 0 (the default): the sharp, cheapest path — same call.
        over_sharp = ui.in_screen_rect(cursor, SHARP_ANCHOR, SHARP_SIZE)
        ui.draw_screen_rect(SHARP_ANCHOR, SHARP_SIZE, YELLOW if over_sharp else PURPLE)

        ui.draw_screen_text(
            60, 50, f"cursor ({cursor[0]:.0f}, {cursor[1]:.0f})", 14.0, LABEL
        )
        ui.draw_screen_text(
            60,
            368,
            f"rect: {over_rect}   circle: {over_circle}   sharp rect: {over_sharp}",
            14.0,
            LABEL,
        )

    app.run(draw_frame)
    return 0


if __name__ == "__main__":
    sys.exit(main())
