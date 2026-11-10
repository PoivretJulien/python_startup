"""Example 6 — sliders that set values.

Four sliders, all in screen space. Three horizontal ones set the red, green
and blue of the swatch on the right; the vertical one sets its corner radius
(in whole pixels: it has a step of 1). Drag a handle, or click anywhere on a
line to bring the handle there.

From the project root (after running the installer in toolchain/):

    .venv/bin/python examples/06_slider/main.py            # reactive (default)
    .venv/bin/python examples/06_slider/main.py --poll     # always redraw

A slider is an object, created once, outside ``draw_frame``::

    red = Slider(start, end, minimum, maximum, value,
                 line_color=..., handle_color=..., hover_color=...)

and one call per frame, which handles the mouse, draws the slider and returns
its value as a float::

    r = red.draw(app)

The slider remembers by itself whether it is being dragged (``red.dragging``),
so the program has no flag to manage.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The drawing library is the graphics/ folder of the project.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "graphics"))

import app3d  # noqa: E402
from widgets2d import Slider  # noqa: E402

LABEL = (0.85, 0.92, 1.00)
MUTED = (0.55, 0.55, 0.62)
TRACK = (0.30, 0.30, 0.36, 1.0)  # the line of every slider
WHITE = (1.00, 1.00, 1.00, 1.0)  # every handle under the cursor

LEFT, RIGHT = 90, 330  # the two ends of the horizontal sliders
SWATCH_ANCHOR, SWATCH_SIZE = (430, 70), (170, 170)


def main() -> int:
    app = app3d.App("example 6 — sliders", size=(680, 330))
    ui = app.aa

    # One slider per colour channel: value from 0 to 1. Each has its own
    # handle colour; fill_color paints the part of the line already "used".
    channels = [
        Slider(
            (LEFT, y), (RIGHT, y), 0.0, 1.0, value,
            line_color=TRACK, handle_color=color, hover_color=WHITE, fill_color=color,
        )
        for y, value, color in (
            (90, 0.58, (0.95, 0.30, 0.30, 1.0)),   # red
            (150, 0.25, (0.35, 0.85, 0.40, 1.0)),  # green
            (210, 0.90, (0.35, 0.55, 1.00, 1.0)),  # blue
        )
    ]
    # A vertical slider is the same object with two other points. The value is
    # `minimum` at the first point: here 0 is at the bottom. step=1 snaps it to
    # whole pixels.
    corner = Slider(
        (640, 240), (640, 70), 0, 85, 28, step=1,
        line_color=TRACK, handle_color=(1.0, 0.55, 0.10, 1.0), hover_color=WHITE,
    )

    def draw_frame() -> None:
        # The whole interaction: one call per slider, one float back.
        r, g, b = (slider.draw(app) for slider in channels)
        radius = corner.draw(app)

        # Use the values.
        ui.draw_screen_rect(SWATCH_ANCHOR, SWATCH_SIZE, (r, g, b, 1.0), radius)

        for name, slider, value in zip("RGB", channels, (r, g, b)):
            y = slider.start[1] + 5
            ui.draw_screen_text(40, y, name, 14.0, LABEL)
            ui.draw_screen_text(RIGHT + 24, y, f"{value:.2f}", 13.0, MUTED)
        ui.draw_screen_text(596, 270, f"radius {radius:.0f}", 13.0, MUTED)

        busy = any(slider.dragging for slider in (*channels, corner))
        ui.draw_screen_text(
            40, 300, "dragging" if busy else "drag a handle, or click a line", 13.0, MUTED
        )

    app.run(draw_frame)
    return 0


if __name__ == "__main__":
    sys.exit(main())
