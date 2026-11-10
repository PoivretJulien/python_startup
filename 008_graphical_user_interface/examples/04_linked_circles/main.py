"""Example 4 — two draggable circles linked by a dashed line.

Everything is in screen space (window pixels, top-left origin). Click a circle
and drag it: the dashed line follows, because it is simply drawn from one
centre to the other on every frame. A circle is yellow under the cursor and
while it is dragged; the line brightens when the cursor is on it, and the
label at its middle gives its length in pixels.

From the project root (after running the installer in toolchain/):

    .venv/bin/python examples/04_linked_circles/main.py            # reactive (default)
    .venv/bin/python examples/04_linked_circles/main.py --poll     # always redraw

What to read: the program keeps only two positions and which circle is being
dragged. Nothing "links" the line to the circles — ``draw_frame`` describes the
whole picture from those few values each time, so the picture cannot disagree
with them.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# The drawing library is the graphics/ folder of the project.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "graphics"))

import app3d  # noqa: E402

PURPLE = (0.58, 0.25, 0.90, 1.0)  # r, g, b, a
YELLOW = (1.00, 0.85, 0.10, 1.0)
ORANGE = (1.00, 0.55, 0.10, 1.0)
GREY = (0.55, 0.55, 0.62, 1.0)
LABEL = (0.85, 0.92, 1.00)

RADIUS = 34  # of both circles, pixels
THICKNESS = 3  # of the line
DASH, SPACING = 12, 8  # length of a dash, and of the gap between two


def main() -> int:
    app = app3d.App("example 4 — two linked circles", size=(640, 400))
    ui = app.aa

    # The whole state of the program: two centres, and the drag in progress.
    circles = [[180.0, 220.0], [460.0, 160.0]]  # lists, so they can be moved
    dragged: int | None = None  # index of the circle being dragged, or None
    grab = (0.0, 0.0)  # cursor-to-centre offset at the moment of the click
    was_down = False

    def draw_frame() -> None:
        nonlocal dragged, grab, was_down

        cursor = app.cursor()
        down = app.mouse_down()

        # Which circle is under the cursor? The last one drawn is on top, so
        # it is asked first.
        hovered = None
        for index in (1, 0):
            if ui.in_screen_circle(cursor, circles[index], RADIUS):
                hovered = index
                break

        # --- the drag: beginning, end, middle ---------------------------
        if down and not was_down and hovered is not None:
            dragged = hovered
            centre = circles[dragged]
            grab = (centre[0] - cursor[0], centre[1] - cursor[1])
        if not down:
            dragged = None
        was_down = down

        if dragged is not None:
            width, height = app.size()
            circles[dragged][0] = min(max(cursor[0] + grab[0], 0.0), width)
            circles[dragged][1] = min(max(cursor[1] + grab[1], 0.0), height)

        # --- the picture: line first, so the circles are drawn over it ----
        a, b = circles
        # A generous hit test (14 px for a 3 px line) makes the line easy to
        # point at; a circle in front of it wins.
        on_line = hovered is None and dragged is None and ui.in_screen_line(cursor, a, b, 14)
        ui.draw_screen_dashed_line(
            a, b, ORANGE if on_line else GREY, THICKNESS, dash=DASH, spacing=SPACING
        )

        for index, centre in enumerate(circles):
            active = index == dragged or (dragged is None and index == hovered)
            ui.draw_screen_circle(centre, RADIUS, YELLOW if active else PURPLE)

        # The length of the link, written at its middle.
        length = math.dist(a, b)
        middle = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        ui.draw_screen_text(middle[0] + 12, middle[1] - 18, f"{length:.0f} px", 13.0, LABEL)

        ui.draw_screen_text(20, 30, "drag a circle: the dashed line follows", 14.0, LABEL)

    app.run(draw_frame)
    return 0


if __name__ == "__main__":
    sys.exit(main())
