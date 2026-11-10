"""Screen-space shapes: rounded rectangles, circles, triangles, lines and their hit tests.

Everything here speaks **window pixels** with the origin at the **top-left** and
y growing down — the convention of ``glfw.get_cursor_pos`` and of
``draw_screen_text``. A shape drawn at a position is therefore hit-tested with
the raw cursor position, no conversion::

    cursor = glfw.get_cursor_pos(window)
    over = screen2d.in_rect(cursor, (40, 60), (220, 90), radius=18)
    system.draw_screen_rect((40, 60), (220, 90), YELLOW if over else PURPLE, 18)

One description, two uses: a rounded rectangle is ``anchor`` (its top-left
corner), ``size`` (width, height) and ``radius`` (the four corners); a circle is
``center`` and ``radius``; a line is ``start``, ``end`` and ``thickness``; a
triangle is ``anchor`` (the middle of its base), ``base``, ``height`` and
``direction`` (where the height points). The draw methods and the hit tests take them in the
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
# One line = one instance row: start(2) end(2) color(4) half_width dash gap.
LINE_FLOATS = 11
# One triangle = one instance row: corner a(2) b(2) c(2) color(4).
TRIANGLE_FLOATS = 10

UP = (0.0, -1.0)  # screen y grows down, so "up" is minus y


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


def line_instance(
    start: Point2,
    end: Point2,
    color: ColorRGBA,
    thickness: float,
    dash: float = 0.0,
    spacing: float = 0.0,
) -> np.ndarray:
    """One line (solid when ``dash`` is 0) as a ``(1, 11)`` float32 instance row."""
    row = np.empty((1, LINE_FLOATS), dtype=np.float32)
    row[0, 0:2] = (float(start[0]), float(start[1]))
    row[0, 2:4] = (float(end[0]), float(end[1]))
    row[0, 4:8] = rgba(color)
    row[0, 8] = 0.5 * max(float(thickness), 0.0)
    row[0, 9] = max(float(dash), 0.0)
    row[0, 10] = max(float(spacing), 0.0)
    return row


def triangle_points(
    anchor: Point2,
    base: float,
    height: float,
    direction: Point2 = UP,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """The three corners of an isosceles triangle: ``(base_a, base_b, tip)``.

    ``anchor`` is the **middle of the base**. ``direction`` is where the height
    points, from that middle to the tip: any ``(x, y)`` (a tuple or a Vec2), of
    any length — only where it points matters. The tip is therefore
    ``anchor + height * unit(direction)`` and the base lies across it,
    ``base`` pixels wide. A zero ``direction`` points up.

    Screen y grows **down**: ``(0, -1)`` points up, ``(1, 0)`` right, ``(0, 1)``
    down, ``(-1, 0)`` left.
    """
    dx, dy = float(direction[0]), float(direction[1])
    length = math.hypot(dx, dy)
    if length <= 1e-12:
        dx, dy = UP
    else:
        dx, dy = dx / length, dy / length
    ax, ay = float(anchor[0]), float(anchor[1])
    half = 0.5 * float(base)
    # (-dy, dx) is the direction turned a quarter turn: the line of the base.
    return (
        (ax - dy * half, ay + dx * half),
        (ax + dy * half, ay - dx * half),
        (ax + dx * float(height), ay + dy * float(height)),
    )


def triangle_instance(
    anchor: Point2,
    base: float,
    height: float,
    direction: Point2,
    color: ColorRGBA,
) -> np.ndarray:
    """One triangle as a ``(1, 10)`` float32 instance row (its three corners)."""
    a, b, c = triangle_points(anchor, base, height, direction)
    row = np.empty((1, TRIANGLE_FLOATS), dtype=np.float32)
    row[0, 0:2] = a
    row[0, 2:4] = b
    row[0, 4:6] = c
    row[0, 6:10] = rgba(color)
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


def line_distance(cursor: Point2, start: Point2, end: Point2) -> float:
    """Distance in pixels from ``cursor`` to the segment ``start``-``end`` (>= 0)."""
    px, py = cursor[0] - start[0], cursor[1] - start[1]
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    # Where the cursor projects on the line, clamped to stay between the ends.
    t = 0.0 if length_sq <= 0.0 else min(max((px * dx + py * dy) / length_sq, 0.0), 1.0)
    return math.hypot(px - t * dx, py - t * dy)


def in_line(cursor: Point2, start: Point2, end: Point2, thickness: float = 2.0) -> bool:
    """True when ``cursor`` is on the line drawn with ``thickness`` (caps included).

    Same description as ``draw_screen_line``. A thin line is a small target:
    pass a larger ``thickness`` than the drawn one to make it easy to grab. For
    a dashed line the gaps count as part of the line.
    """
    return line_distance(cursor, start, end) <= 0.5 * thickness


def in_triangle(
    cursor: Point2,
    anchor: Point2,
    base: float,
    height: float,
    direction: Point2 = UP,
) -> bool:
    """True when ``cursor`` is inside the triangle (edges included).

    Same description as ``draw_screen_triangle``. The point is inside when it
    is on the same side of the three edges.
    """
    a, b, c = triangle_points(anchor, base, height, direction)

    def side(p, q) -> float:  # > 0 on one side of the edge p-q, < 0 on the other
        return (cursor[0] - q[0]) * (p[1] - q[1]) - (p[0] - q[0]) * (cursor[1] - q[1])

    d1, d2, d3 = side(a, b), side(b, c), side(c, a)
    negative = d1 < 0.0 or d2 < 0.0 or d3 < 0.0
    positive = d1 > 0.0 or d2 > 0.0 or d3 > 0.0
    return not (negative and positive)


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
