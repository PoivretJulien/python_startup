"""Screen-space shapes: rounded rectangles, circles and their hit tests.

Everything here speaks **window pixels** with the origin at the **top-left** and
y growing down — the convention of ``glfw.get_cursor_pos`` and of
``draw_screen_text``. A shape drawn at a position is therefore hit-tested with
the raw cursor position, no conversion::

    cursor = glfw.get_cursor_pos(window)
    over = screen2d.in_rect(cursor, (40, 60), (220, 90), radius=18)
    system.draw_screen_rect((40, 60), (220, 90), YELLOW if over else PURPLE, 18)

One description, two uses: a rounded rectangle is ``anchor`` (its top-left
corner), ``size`` (width, height) and ``radius`` (the four corners); a circle is
``center`` and ``radius``. The draw methods and the hit tests take them in the
same order, so the picture and the test cannot drift apart.

The hit tests are the same signed distance function (SDF) the fragment shader
evaluates (``shaders3d.SCREEN_SHAPES``), run once on the CPU for one point:

    q = |p - center| - half_size + radius
    d = length(max(q, 0)) + min(max(q.x, q.y), 0) - radius      # d <= 0: inside

``radius == 0`` never reaches that formula: a sharp rectangle is four
comparisons here and a plain solid quad on the GPU.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Point2 = Sequence[float]  # (x, y) — a tuple, a list or a Vec2
Size2 = Sequence[float]  # (width, height)
ColorRGBA = Sequence[float]  # (r, g, b) or (r, g, b, a), each 0..1

# One shape = one instance row: center(2) half_size(2) color(4) radius(1).
SHAPE_FLOATS = 9


def clamp_radius(width: float, height: float, radius: float) -> float:
    """A corner radius can never exceed half of the smaller side."""
    return max(0.0, min(float(radius), 0.5 * abs(width), 0.5 * abs(height)))


def rgba(color: ColorRGBA) -> tuple[float, float, float, float]:
    """``(r, g, b)`` or ``(r, g, b, a)`` -> four floats (alpha defaults to 1)."""
    alpha = float(color[3]) if len(color) > 3 else 1.0
    return float(color[0]), float(color[1]), float(color[2]), alpha


# ----------------------------------------------------------------- packing
def rect_instance(
    anchor: Point2,
    size: Size2,
    color: ColorRGBA,
    radius: float = 0.0,
) -> np.ndarray:
    """One rounded rectangle as a ``(1, 9)`` float32 instance row.

    A negative width or height simply grows the other way from ``anchor``.
    """
    width, height = float(size[0]), float(size[1])
    row = np.empty((1, SHAPE_FLOATS), dtype=np.float32)
    row[0, 0] = float(anchor[0]) + 0.5 * width
    row[0, 1] = float(anchor[1]) + 0.5 * height
    row[0, 2] = 0.5 * abs(width)
    row[0, 3] = 0.5 * abs(height)
    row[0, 4:8] = rgba(color)
    row[0, 8] = clamp_radius(width, height, radius)
    return row


def circle_instance(center: Point2, radius: float, color: ColorRGBA) -> np.ndarray:
    """One disc: a square whose corner radius is its half size."""
    radius = abs(float(radius))
    row = np.empty((1, SHAPE_FLOATS), dtype=np.float32)
    row[0, 0] = float(center[0])
    row[0, 1] = float(center[1])
    row[0, 2] = row[0, 3] = row[0, 8] = radius
    row[0, 4:8] = rgba(color)
    return row


def fit(content: Size2, anchor: Point2, size: Size2) -> tuple[tuple[float, float], tuple[float, float]]:
    """The largest ``(anchor, size)`` with ``content``'s proportions inside a box.

    ``content`` is ``(width, height)`` of a picture (only its ratio matters);
    ``anchor`` / ``size`` describe the box it must stay in. The result is
    centred in the box and is ready for ``draw_screen_image``::

        where, how_big = screen2d.fit((640, 480), (0, 0), app.size())
        ui.draw_screen_image(where, how_big, picture)
    """
    box_w, box_h = abs(float(size[0])), abs(float(size[1]))
    ratio = float(content[0]) / max(float(content[1]), 1e-9)
    width = min(box_w, box_h * ratio)
    height = width / max(ratio, 1e-9)
    x = float(anchor[0]) + 0.5 * (box_w - width)
    y = float(anchor[1]) + 0.5 * (box_h - height)
    return (x, y), (width, height)


# --------------------------------------------------------------- hit tests
def in_rect(
    cursor: Point2,
    anchor: Point2,
    size: Size2,
    radius: float = 0.0,
) -> bool:
    """True when ``cursor`` is inside the rounded rectangle (edge included).

    Same description as ``draw_screen_rect``: ``anchor`` is the top-left
    corner, ``size`` is ``(width, height)``, ``radius`` rounds the four corners
    (0 = sharp, which is four comparisons and nothing else).
    """
    half_w, half_h = 0.5 * abs(size[0]), 0.5 * abs(size[1])
    # Fold the point into the first quadrant around the centre.
    dx = abs(cursor[0] - (anchor[0] + 0.5 * size[0]))
    dy = abs(cursor[1] - (anchor[1] + 0.5 * size[1]))
    if dx > half_w or dy > half_h:
        return False  # outside the bounding box: the common, cheap answer
    radius = min(max(radius, 0.0), half_w, half_h)
    # Offset from the centre of the corner arc; only a point beyond it on both
    # axes sits in the corner square, where the arc decides.
    qx, qy = dx - (half_w - radius), dy - (half_h - radius)
    if qx <= 0.0 or qy <= 0.0:
        return True
    return qx * qx + qy * qy <= radius * radius


def in_circle(cursor: Point2, center: Point2, radius: float) -> bool:
    """True when ``cursor`` is inside the circle (edge included)."""
    dx, dy = cursor[0] - center[0], cursor[1] - center[1]
    return dx * dx + dy * dy <= radius * radius


def rect_distance(
    cursor: Point2,
    anchor: Point2,
    size: Size2,
    radius: float = 0.0,
) -> float:
    """Signed distance in pixels from ``cursor`` to the rounded rectangle's edge.

    Negative inside, zero on the edge, positive outside — the exact function the
    fragment shader fades over one pixel. Handy for a hover margin
    (``rect_distance(...) <= 4``) or a glow.
    """
    half_w, half_h = 0.5 * abs(size[0]), 0.5 * abs(size[1])
    radius = min(max(radius, 0.0), half_w, half_h)
    qx = abs(cursor[0] - (anchor[0] + 0.5 * size[0])) - half_w + radius
    qy = abs(cursor[1] - (anchor[1] + 0.5 * size[1])) - half_h + radius
    return math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0) - radius


def circle_distance(cursor: Point2, center: Point2, radius: float) -> float:
    """Signed distance in pixels from ``cursor`` to the circle's edge."""
    return math.hypot(cursor[0] - center[0], cursor[1] - center[1]) - radius
