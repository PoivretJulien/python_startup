"""Pure 3D stroke geometry, the adaptive floor grid, and the grid-step ladder.

No GLFW or OpenGL imports here — like ``experimental/geometry.py`` this module
is only math. A batch is the same layout the plain renderer uploads as
``GL_LINES``::

    x, y, z, r, g, b     (6 float32 per vertex)

Draw helpers take plain 3-tuples, so the public API stays as small as the 2D
one: ``draw_arrow(start, tip, color)``, ``draw_line(start, tip, color)``,
``draw_circle(center, radius, color, axis)``.

A circle in 3D needs one extra piece of information the 2D version did not: the
plane it lives in. That is what ``axis`` is — the circle's normal (the axis you
would spin it around), not a segment drawn through its centre.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from settings3d import (
    ARROW_HEAD_ANGLE,
    ARROW_HEAD_LENGTH,
    AXIS_X_COLOR,
    AXIS_Y_COLOR,
    AXIS_Z_COLOR,
    GRID_COLOR,
    GROUND_Z,
    STEP,
    TAIL_MARKER,
)

Color = tuple[float, float, float]
Vector3 = Sequence[float] | np.ndarray
# Tuples, lists, NumPy arrays and sequence-like vector classes all work.
Point3 = Vector3

_X = np.array((1.0, 0.0, 0.0), dtype=np.float64)
_Y = np.array((0.0, 1.0, 0.0), dtype=np.float64)
_Z = np.array((0.0, 0.0, 1.0), dtype=np.float64)


# --------------------------------------------------------------------- step ladder
def axis_range(low: float, high: float, step: float) -> tuple[int, int]:
    """Inclusive index range of the grid lines ``k * step`` inside ``low..high``."""
    eps = 1e-9
    return math.ceil(low / step - eps), math.floor(high / step + eps)


def nice_step_at_most(maximum: float) -> float:
    """Largest of 1, 2, 5 x 10^k that is <= `maximum` (a human grid spacing).

    Rounding *down* keeps one world unit as one square at the initial framing:
    the grid only subdivides when the camera gets close enough that a unit is
    larger than the pixel budget.
    """
    if maximum <= 0.0:
        return STEP
    exponent = math.floor(math.log10(maximum))
    base = maximum / (10.0**exponent)
    for candidate in (5.0, 2.0, 1.0):
        if candidate <= base + 1e-9:
            return candidate * 10.0**exponent
    return 10.0**exponent


def next_nice_step(step: float) -> float:
    """Next coarser value on the 1/2/5 x 10^k ladder (1 -> 2 -> 5 -> 10)."""
    if step <= 0.0:
        return STEP
    exponent = math.floor(math.log10(step) + 1e-9)
    mantissa = step / 10.0**exponent
    if mantissa < 1.5:
        return 2.0 * 10.0**exponent
    if mantissa < 3.5:
        return 5.0 * 10.0**exponent
    return 10.0 ** (exponent + 1)


# -------------------------------------------------------------------------- grid
def grid_line_count(center_x: float, center_y: float, extent: float, step: float) -> int:
    """Line segments :func:`grid_vertices` emits for this camera + step."""
    first_x, last_x = axis_range(center_x - extent, center_x + extent, step)
    first_y, last_y = axis_range(center_y - extent, center_y + extent, step)
    xs = max(0, last_x - first_x + 1)
    ys = max(0, last_y - first_y + 1)
    return xs + ys + 1  # + the vertical z axis


def grid_vertices(
    center_x: float,
    center_y: float,
    extent: float,
    step: float = STEP,
    *,
    ground_z: float = GROUND_Z,
) -> np.ndarray:
    """Floor grid on ``z = ground_z`` plus the three world axes (z is up).

    The grid is rebuilt from the *target-centred* extent, and its lines sit on
    world-space multiples of ``step`` — so panning slides the grid across the
    floor instead of dragging a screen-stuck one. x and y are one colour each;
    the vertical z axis is a third, so an orbited camera can still tell which
    direction is which.
    """
    step = step if step > 0.0 else STEP
    rows: list[tuple[float, ...]] = []

    first, last = axis_range(center_x - extent, center_x + extent, step)
    for i in range(first, last + 1):
        x = i * step
        # A line at x = 0 runs along y: that is the y axis.
        color = AXIS_Y_COLOR if i == 0 else GRID_COLOR
        rows += [
            (x, center_y - extent, ground_z, *color),
            (x, center_y + extent, ground_z, *color),
        ]

    first, last = axis_range(center_y - extent, center_y + extent, step)
    for j in range(first, last + 1):
        y = j * step
        # A line at y = 0 runs along x: that is the x axis.
        color = AXIS_X_COLOR if j == 0 else GRID_COLOR
        rows += [
            (center_x - extent, y, ground_z, *color),
            (center_x + extent, y, ground_z, *color),
        ]

    # The only vertical line in the scene: the z axis.
    rows += [
        (0.0, 0.0, -extent, *AXIS_Z_COLOR),
        (0.0, 0.0, extent, *AXIS_Z_COLOR),
    ]
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


# ----------------------------------------------------------------------- circles
def _normalized(vector: Vector3) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    length = math.sqrt(float(v @ v))
    return v / length if length > 1e-12 else np.zeros(3)


def basis_from_axis(axis: Vector3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An orthonormal ``(u, v, n)`` frame with ``n`` along ``axis``.

    Anything perpendicular to a circle's axis works; picking the world axis the
    normal is *least* aligned with keeps the frame stable for every direction,
    including the usual (0, 0, 1) and (0, 1, 0).
    """
    n = _normalized(axis)
    if float(n @ n) < 1e-12:
        n = _Z.copy()
    helper = min((_X, _Y, _Z), key=lambda candidate: abs(float(n @ candidate)))
    u = _normalized(np.cross(n, helper))
    v = np.cross(n, u)
    return u, v, n


def circle_vertices(
    center: Point3,
    radius: float,
    *,
    color: Color,
    axis: Vector3 = (0.0, 0.0, 1.0),
    segments: int = 64,
) -> np.ndarray:
    """A closed circle outline in the plane whose normal is ``axis``.

    ``axis`` is the spinning axis: ``(0, 0, 1)`` lies flat on the floor,
    ``(0, 1, 0)`` stands in the xz plane, ``(1, 0, 0)`` stands in the yz plane.
    """
    c = np.asarray(center, dtype=np.float64)
    u, v, _n = basis_from_axis(axis)
    segments = max(3, int(segments))
    rows: list[tuple[float, ...]] = []
    for i in range(segments):
        a = math.tau * i / segments
        b = math.tau * (i + 1) / segments
        pa = c + (u * math.cos(a) + v * math.sin(a)) * radius
        pb = c + (u * math.cos(b) + v * math.sin(b)) * radius
        rows += [
            (pa[0], pa[1], pa[2], *color),
            (pb[0], pb[1], pb[2], *color),
        ]
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


# ------------------------------------------------------------------------ arrows
def _screen_frame(direction: np.ndarray, view_dir: Vector3) -> tuple[np.ndarray, np.ndarray]:
    """Screen-plane ``(along, across)`` unit vectors for a 3D direction.

    The arrow head is a decoration, not a solid: it should always be readable.
    Flattening the segment onto the camera plane and opening the head in that
    plane makes it face the viewer from any orbit angle, which a fixed world
    plane would not.
    """
    view = _normalized(view_dir)
    along = direction - view * float(direction @ view)
    if float(along @ along) < 1e-12:
        # The arrow points straight at (or away from) the camera. Any
        # perpendicular is as good as another; prefer one that is not parallel
        # to the direction to keep the cross product below well conditioned.
        helper = (
            np.array((0.0, 1.0, 0.0))
            if abs(float(direction[1])) < 0.9
            else np.array((1.0, 0.0, 0.0))
        )
        along = np.cross(direction, helper)
    along = _normalized(along)
    across = _normalized(np.cross(view, along))
    return along, across


def line_vertices(
    start: Point3,
    tip: Point3,
    *,
    color: Color,
) -> np.ndarray:
    """A plain segment: no arrow head and no tail marker."""
    rows = [(*start, *color), (*tip, *color)]
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def arrow_vertices(
    start: Point3,
    tip: Point3,
    *,
    view_dir: Vector3,
    color: Color,
    head_length: float = ARROW_HEAD_LENGTH,
    head_angle_degrees: float = ARROW_HEAD_ANGLE,
    tail_marker: float = TAIL_MARKER,
) -> np.ndarray:
    """Shaft + two readable head strokes + a screen-facing cross at the tail.

    ``view_dir`` is the camera's forward vector (eye -> target). It only affects
    the *decoration*: the shaft is the exact 3D segment, while the head and tail
    cross are laid out in the plane of the screen so they never disappear
    edge-on when the camera orbits.
    """
    start_array = np.asarray(start, dtype=np.float64)
    tip_array = np.asarray(tip, dtype=np.float64)
    delta = tip_array - start_array
    length = math.sqrt(float(delta @ delta))

    direction = _normalized(delta) if length > 1e-9 else _normalized((1.0, 0.0, 0.0))
    along, across = _screen_frame(direction, view_dir)

    rows: list[tuple[float, ...]] = [(*start, *color), (*tip, *color)]

    if length > 1e-6:
        head_length = min(head_length, length * 0.45)
        angle = math.radians(head_angle_degrees)
        for side in (-1.0, 1.0):
            wing = (-along * math.cos(angle) + across * (math.sin(angle) * side)) * head_length
            end = tip_array + wing
            rows += [(*tip, *color), (end[0], end[1], end[2], *color)]

    if tail_marker > 0.0:
        for axis in (along, across):
            a = start_array - axis * tail_marker
            b = start_array + axis * tail_marker
            rows += [
                (a[0], a[1], a[2], *color),
                (b[0], b[1], b[2], *color),
            ]

    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def thick_line_vertices(
    start: Point3,
    tip: Point3,
    thickness: float,
    *,
    view_dir: Vector3,
    color: Color,
) -> np.ndarray:
    """A segment expanded into a camera-facing ribbon (two triangles).

    Only the *plain* renderer needs this: its GL_LINES path cannot set a width,
    so a world-space ``thickness`` is approximated by a billboard quad. The AA
    renderer has real pixel widths and measures its distance field instead.
    """
    start_array = np.asarray(start, dtype=np.float64)
    tip_array = np.asarray(tip, dtype=np.float64)
    delta = tip_array - start_array
    length = math.sqrt(float(delta @ delta))
    if length < 1e-9 or thickness <= 0.0:
        # Degenerate segment: six identical vertices draw nothing.
        rows = [(*start, *color)] * 6
        return np.array(rows, dtype=np.float32).reshape(-1, 6)

    along, across = _screen_frame(delta / length, view_dir)
    half = across * (thickness * 0.5)
    a = start_array + half
    b = start_array - half
    c = tip_array + half
    d = tip_array - half
    rows = [
        (a[0], a[1], a[2], *color),
        (b[0], b[1], b[2], *color),
        (d[0], d[1], d[2], *color),
        (a[0], a[1], a[2], *color),
        (d[0], d[1], d[2], *color),
        (c[0], c[1], c[2], *color),
    ]
    return np.array(rows, dtype=np.float32).reshape(-1, 6)
