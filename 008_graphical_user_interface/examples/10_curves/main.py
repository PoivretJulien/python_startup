"""Example 10 — polylines and NURBS curves.

Two things on screen:

* in front, a NURBS curve you can edit: drag its six control points with the
  mouse. The thin grey line through them is a polyline;
* behind, in 3D, a spiral made of 20 000 points drawn by ONE call, an exact
  circle made by a NURBS, and a pink NURBS that climbs through space. Click
  one of its grey control points: a gizmo appears on it. Drag an arrow to
  slide the point along x, y or z, a small square to move it in a plane
  (two axes at once), or the point itself to move it freely;
* the text sits on a panel of frosted glass ("acrylic") that blurs the scene
  behind it.

From the project root (after running the installer in toolchain/):

    .venv/bin/python examples/10_curves/main.py
    .venv/bin/python examples/10_curves/main.py --points 200000    # a longer spiral

Keys:

    Up / Down     the degree of the curve (1 = straight segments, 3 = smooth)
    W / S         the weight of the control point you touched last
                  (heavier = the curve comes closer to it)
    C             open / closed curve
    Left / Right  move a yellow marker along the pink curve: it is drawn at
                  ``ribbon.point(t)``, with t going from 0 (start) to 1 (end)

The camera: left-drag on the background turns the 3D scene, right-drag (or two
fingers on a trackpad) slides it, the wheel or a pinch zooms.

The two ideas:

* a polyline is one array of points and one call — ``draw_polyline(points,
  color)`` — never a loop of ``draw_line``;
* a NURBS is a few control points; the library computes the points of the
  smooth curve and draws them as a polyline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# The drawing library is the graphics/ folder of the project.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "graphics"))

import app3d  # noqa: E402
from curve3d import Nurbs  # noqa: E402

PURPLE = (0.58, 0.25, 0.90, 1.0)
YELLOW = (1.00, 0.85, 0.10, 1.0)
ORANGE = (1.00, 0.55, 0.10, 1.0)
GREY = (0.55, 0.55, 0.62, 0.8)
CYAN = (0.20, 0.85, 0.95)
PINK = (1.00, 0.35, 0.70)
GLASS = (0.10, 0.10, 0.14, 0.45)  # the tint of the acrylic panel: r, g, b, strength
LABEL = (0.85, 0.92, 1.00)
MUTED = (0.60, 0.60, 0.68)

HANDLE_RADIUS = 9


def make_spiral(count: int) -> tuple[np.ndarray, np.ndarray]:
    """``count`` points winding up a cone, and one colour per point."""
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)
    turns = 60.0 * np.pi * t
    radius = 3.0 * (1.0 - 0.7 * t)
    points = np.stack((radius * np.cos(turns), radius * np.sin(turns), 3.0 * t), axis=1)
    colors = np.stack((0.2 + 0.8 * t, 0.9 - 0.5 * t, 1.0 - 0.8 * t), axis=1)  # blue to orange
    return points, colors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--points", type=int, default=20_000, help="points in the 3D spiral")
    app = app3d.App("Example 10 — curves", size=(900, 620), parser=parser)
    ui = app.aa
    app.camera.distance = 9.0

    # Built once: the arrays do not change, only the camera does.
    spiral, spiral_colors = make_spiral(max(app.args.points, 2))
    circle = Nurbs.circle((0, 0, 0), 3.0)  # an exact circle, on the floor
    # A curve in space: every control point has its own height (z).
    ribbon = Nurbs(
        [(-4, -3, 0.2), (-2, 3, 1.5), (0, -3, 3.0), (2, 3, 1.5), (4, -3, 0.2), (4, 2, 3.5), (0, 4, 4.0)]
    )

    # The curve to edit: control points in window pixels.
    curve = Nurbs([(90, 470), (220, 250), (380, 520), (520, 230), (680, 520), (810, 300)])
    marker_t = 0.5  # where the marker is on the pink curve: 0 = start, 1 = end
    grabbed = -1  # index of the control point being dragged, -1 = none
    selected = 1  # the one W / S act on
    was_down = False

    def draw_frame() -> None:
        nonlocal grabbed, selected, was_down, marker_t
        cursor = app.cursor()
        down = app.mouse_down()
        pressed = down and not was_down
        was_down = down

        # --- drag a control point ------------------------------------------
        hovered = -1
        for index, point in enumerate(curve.control_points):
            if ui.in_screen_circle(cursor, point, HANDLE_RADIUS + 4):
                hovered = index
        if pressed and hovered >= 0:
            grabbed = selected = hovered
        if not down:
            grabbed = -1
        if grabbed >= 0:
            curve.control_points[grabbed] = (float(cursor[0]), float(cursor[1]))
        elif not (down and hovered >= 0) and not ribbon.busy:
            app.handle_camera()  # the mouse is free: it moves the 3D view

        # --- the keyboard --------------------------------------------------
        for key in app.keys_pressed(repeat=True):
            if key == "UP":
                curve.degree = min(curve.degree + 1, len(curve.control_points) - 1)
            elif key == "DOWN":
                curve.degree = max(curve.degree - 1, 1)
            elif key == "W":
                curve.weights[selected] = min(curve.weights[selected] * 1.25, 50.0)
            elif key == "S":
                curve.weights[selected] = max(curve.weights[selected] / 1.25, 0.05)
            elif key == "C":
                curve.closed = not curve.closed
            elif key == "RIGHT":
                marker_t = min(marker_t + 0.02, 1.0)
            elif key == "LEFT":
                marker_t = max(marker_t - 0.02, 0.0)

        # --- the 3D scene: one call per curve ------------------------------
        ui.draw_grid()
        ui.draw_polyline(spiral, spiral_colors)  # 20 000 points, one draw call
        ui.draw_nurbs(circle, CYAN, thickness=0.05, samples=96)
        # pick=True: its control points can be clicked and moved with a gizmo.
        ui.draw_nurbs(ribbon, PINK, thickness=0.06, samples=200, pick=True)

        # A point ON the curve: point(t) turns a number from 0 to 1 into (x, y, z).
        # It follows the curve when its control points are moved.
        marker = ribbon.point(marker_t)
        ui.draw_point(marker, 0.14, YELLOW[:3])
        spot = ui.world_to_screen(marker)  # where it is on screen, for its label
        if spot is not None:
            ui.draw_screen_text(
                spot[0] + 14, spot[1] - 10,
                f"t = {marker_t:.2f}   ({marker[0]:.2f}, {marker[1]:.2f}, {marker[2]:.2f})", 12, YELLOW,
            )

        # --- the editable curve, on the screen -----------------------------
        ui.draw_screen_polyline(curve.control_points, GREY, 1, closed=curve.closed)
        ui.draw_screen_nurbs(curve, ORANGE, 4, samples=160)
        for index, point in enumerate(curve.control_points):
            lit = index == grabbed or (index == hovered and not down)
            ui.draw_screen_circle(point, HANDLE_RADIUS, YELLOW if lit else PURPLE)
            if index == selected:
                ui.draw_screen_circle_outline(point, HANDLE_RADIUS + 4, YELLOW, 1.5)

        # A frosted panel under the text: drawn after the scene (it blurs what
        # is already there), before the text (which stays sharp on top).
        ui.draw_screen_acrylic_rect((8, 6), (884, 54), GLASS, radius=12, blur=14)
        ui.draw_screen_text(
            16, 26,
            f"degree {curve.degree}   weight of point {selected + 1}: {curve.weights[selected]:.2f}"
            f"   {'closed' if curve.closed else 'open'}   spiral: {len(spiral):,} points",
            14, LABEL,
        )
        ui.draw_screen_text(
            16, 48, "drag the points   click a grey point: 3D gizmo"
            "   Left/Right marker   Up/Down degree   W/S weight   C open/closed", 12, MUTED,
        )

    app.run(draw_frame)


if __name__ == "__main__":
    main()
