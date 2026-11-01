"""Pure 2D geometry, the camera, and the screen <-> world mapping.

No GLFW or OpenGL imports here, so the math can be tested or reused on its own.
The grid is a batch of GL_LINES vertices: ``(x, y, z, r, g, b)`` per vertex.

The camera is the single source of truth for the *domain*: the same object
answers "which world rectangle is visible", "how do world units map to the
clip space the vertex shaders expect", and "which world point is under this
pixel". Cursor feedback, panning, zooming and the grid all read it, so they can
never disagree — which is exactly what broke when the domain started moving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from settings import (
    ARROW_COLOR,
    ARROW_HEAD_ANGLE,
    ARROW_HEAD_LENGTH,
    AXIS_COLOR,
    CAMERA_MAX_HALF,
    CAMERA_MIN_HALF,
    GRID_COLOR,
    GRID_MAX_LINES,
    GRID_MAX_SPACING_PX,
    HALF,
    STEP,
    TAIL_MARKER,
)

Point = tuple[float, float]
Color = tuple[float, float, float]


def axis_range(low: float, high: float, step: float) -> tuple[int, int]:
    """Inclusive index range of the grid lines ``k * step`` inside ``low..high``.

    An epsilon makes an exact boundary inclusive instead of dropping the last line.
    """
    eps = 1e-9
    return math.ceil(low / step - eps), math.floor(high / step + eps)


def grid_line_count(min_x: float, min_y: float, max_x: float, max_y: float, step: float) -> int:
    """How many line segments :func:`grid_vertices` would emit for this view."""
    first_x, last_x = axis_range(min_x, max_x, step)
    first_y, last_y = axis_range(min_y, max_y, step)
    return max(0, last_x - first_x + 1) + max(0, last_y - first_y + 1)


def grid_vertices(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    step: float = STEP,
    *,
    grid_color: Color = GRID_COLOR,
    axis_color: Color = AXIS_COLOR,
) -> np.ndarray:
    """Grid lines covering the visible world rectangle ``(min_x, min_y, max_x, max_y)``.

    The domain is no longer a fixed +/-HALF square: it is whatever the camera
    shows, so the grid is rebuilt from the visible rectangle and always fills the
    window. Lines sit on multiples of ``step`` in *world* space, so panning slides
    them across the screen instead of dragging a screen-stuck grid around; the
    axis lines at 0 stay orange whenever they are in view.
    """
    step = step if step > 0.0 else STEP
    rows: list[tuple[float, ...]] = []

    first, last = axis_range(min_x, max_x, step)
    for i in range(first, last + 1):
        x = i * step
        color = axis_color if i == 0 else grid_color
        rows += [(x, min_y, 0.0, *color), (x, max_y, 0.0, *color)]

    first, last = axis_range(min_y, max_y, step)
    for j in range(first, last + 1):
        y = j * step
        color = axis_color if j == 0 else grid_color
        rows += [(min_x, y, 0.0, *color), (max_x, y, 0.0, *color)]

    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def arrow_vertices(
    start: Point,
    tip: Point,
    *,
    color: Color = ARROW_COLOR,
    head_length: float = ARROW_HEAD_LENGTH,
    head_angle_degrees: float = ARROW_HEAD_ANGLE,
    tail_marker: float = TAIL_MARKER,
) -> np.ndarray:
    """Shaft + two head strokes + a small cross marking the tail."""
    sx, sy = start
    tx, ty = tip
    rows: list[tuple[float, ...]] = [(*start, 0.0, *color), (*tip, 0.0, *color)]

    dx, dy = tx - sx, ty - sy
    length = math.hypot(dx, dy)
    if length > 1e-6:
        direction = math.atan2(dy, dx)
        head_length = min(head_length, length * 0.45)
        head_angle = math.radians(head_angle_degrees)
        for side in (-1, 1):
            angle = direction + math.pi + side * head_angle
            rows += [
                (tx, ty, 0.0, *color),
                (
                    tx + math.cos(angle) * head_length,
                    ty + math.sin(angle) * head_length,
                    0.0,
                    *color,
                ),
            ]

    if tail_marker > 0.0:
        rows += [
            (sx - tail_marker, sy, 0.0, *color),
            (sx + tail_marker, sy, 0.0, *color),
            (sx, sy - tail_marker, 0.0, *color),
            (sx, sy + tail_marker, 0.0, *color),
        ]
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def line_vertices(
    start: Point,
    tip: Point,
    *,
    color: Color = ARROW_COLOR,
) -> np.ndarray:
    """A plain segment: no arrow head and no tail marker.

    The whole segment is drawn at one scale, so keeping the head proportional to
    world space is up to the caller; this helper exists for the grid/markers.
    """
    rows = [(*start, 0.0, *color), (*tip, 0.0, *color)]
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def thick_line_vertices(
    start: Point,
    tip: Point,
    thickness: float,
    *,
    color: Color = ARROW_COLOR,
) -> np.ndarray:
    """A line segment expanded into a quad (two triangles).

    ``thickness`` is in world units, the same units as ``start`` and ``tip``.
    """
    sx, sy = start
    tx, ty = tip
    dx, dy = tx - sx, ty - sy
    length = math.hypot(dx, dy)
    if length < 1e-9 or thickness <= 0.0:
        # Degenerate segment: six identical vertices draw nothing.
        rows = [(*start, 0.0, *color)] * 6
        return np.array(rows, dtype=np.float32).reshape(-1, 6)

    nx, ny = -dy / length, dx / length
    half = thickness * 0.5
    a = (sx + nx * half, sy + ny * half, 0.0, *color)
    b = (sx - nx * half, sy - ny * half, 0.0, *color)
    c = (tx + nx * half, ty + ny * half, 0.0, *color)
    d = (tx - nx * half, ty - ny * half, 0.0, *color)
    rows = [a, b, d, a, d, c]
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def circle_vertices(
    center: Point,
    radius: float,
    *,
    color: Color = ARROW_COLOR,
    segments: int = 64,
) -> np.ndarray:
    """A closed circle outline, as GL_LINES pairs between consecutive points."""
    cx, cy = center
    segments = max(3, int(segments))
    rows: list[tuple[float, ...]] = []
    for i in range(segments):
        a = math.tau * i / segments
        b = math.tau * (i + 1) / segments
        rows += [
            (cx + radius * math.cos(a), cy + radius * math.sin(a), 0.0, *color),
            (cx + radius * math.cos(b), cy + radius * math.sin(b), 0.0, *color),
        ]
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def nice_step_at_most(maximum: float) -> float:
    """Largest of 1, 2, 5 x 10^k that is <= `maximum` (a "human" grid spacing).

    Rounding *down* is what keeps the 1-unit grid at 1:1 zoom: a step is only
    subdivided once one world unit covers more pixels than the budget allows.
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
    """The next coarser value on the 1/2/5 x 10^k ladder (1 -> 2 -> 5 -> 10).

    Used only by the grid line-count safety cap, so it coarsens by the smallest
    readable jump instead of multiplying by ten.
    """
    if step <= 0.0:
        return STEP
    exponent = math.floor(math.log10(step) + 1e-9)
    mantissa = step / 10.0**exponent
    if mantissa < 1.5:
        return 2.0 * 10.0**exponent
    if mantissa < 3.5:
        return 5.0 * 10.0**exponent
    return 10.0 ** (exponent + 1)


@dataclass
class Camera:
    """A 2D orthographic camera: where it looks, and how much of the world it sees.

    ``center`` is the world point at the middle of the viewport and
    ``view_half_y`` is how many world units fit between that point and the top
    edge. One number is enough: the x half-extent follows from the window aspect
    ratio, so world units stay square at every window size and zoom level.

    Zoom is therefore *inverse* to ``view_half_y``: zooming in shrinks the world
    window (``view_half_y`` goes down), zooming out grows it. ``HALF / view_half_y``
    is the zoom factor shown in the readout, where 1.0 is the initial view.
    """

    center: Point = (0.0, 0.0)
    view_half_y: float = HALF

    # -- view description --------------------------------------------------

    def reset(self) -> None:
        """Back to the startup framing: world origin centred, 1:1 zoom."""
        self.center = (0.0, 0.0)
        self.view_half_y = HALF

    def half_extents(self, width: float, height: float) -> tuple[float, float]:
        """World units from the centre to the right/top edge of the viewport."""
        aspect = max(width, 1.0) / max(height, 1.0)
        if aspect >= 1.0:
            return self.view_half_y * aspect, self.view_half_y
        return self.view_half_y, self.view_half_y / aspect

    def visible_rect(self, width: float, height: float) -> tuple[float, float, float, float]:
        """The world rectangle currently on screen: ``(min_x, min_y, max_x, max_y)``."""
        half_x, half_y = self.half_extents(width, height)
        cx, cy = self.center
        return cx - half_x, cy - half_y, cx + half_x, cy + half_y

    def zoom_factor(self) -> float:
        """1.0 at the initial framing; >1 zoomed in, <1 zoomed out."""
        return HALF / self.view_half_y

    def world_units_per_pixel(self, height: float) -> float:
        """How much world space one pixel covers vertically (and horizontally)."""
        return 2.0 * self.view_half_y / max(height, 1.0)

    # -- world <-> screen --------------------------------------------------

    def world_to_clip(self, width: float, height: float) -> tuple[float, float, float, float]:
        """``(scale_x, scale_y, offset_x, offset_y)``: ``clip = world * scale + offset``.

        This is the whole vertex shader camera. It bakes the aspect correction,
        the zoom and the pan into one vec4 upload.
        """
        half_x, half_y = self.half_extents(width, height)
        scale_x, scale_y = 1.0 / half_x, 1.0 / half_y
        cx, cy = self.center
        return scale_x, scale_y, -cx * scale_x, -cy * scale_y

    def screen_to_world(
        self,
        px: float,
        py: float,
        width: float,
        height: float,
    ) -> Point:
        """Window cursor (pixels, y down) -> world (units, y up).

        Exact inverse of :meth:`world_to_clip`, which is what keeps the clamped
        arrow tip glued to the real cursor at every pan/zoom level.
        """
        half_x, half_y = self.half_extents(width, height)
        cx, cy = self.center
        ndc_x = 2.0 * px / max(width, 1.0) - 1.0
        ndc_y = 1.0 - 2.0 * py / max(height, 1.0)
        return cx + ndc_x * half_x, cy + ndc_y * half_y

    # -- navigation --------------------------------------------------------

    def pan_pixels(self, dx_px: float, dy_px: float, width: float, height: float) -> None:
        """Drag the *content* by (dx, dy) window pixels (y down), like grabbing it.

        The camera centre moves the other way, in world units, so the scene stays
        exactly under the pointer at any zoom level.
        """
        half_x, half_y = self.half_extents(width, height)
        cx, cy = self.center
        self.center = (
            cx - dx_px * 2.0 * half_x / max(width, 1.0),
            cy + dy_px * 2.0 * half_y / max(height, 1.0),
        )

    def zoom_at(
        self,
        px: float,
        py: float,
        factor: float,
        width: float,
        height: float,
    ) -> None:
        """Scale the view by `factor` while keeping the world under (px, py) fixed.

        That is the behaviour that makes wheel/pinch zoom feel anchored to the
        pointer instead of to the window centre: the point under the cursor is
        sampled before and after the zoom, and the centre is corrected by the
        difference.
        """
        if factor <= 0.0 or factor == 1.0:
            return
        before = self.screen_to_world(px, py, width, height)
        self.zoom_center(factor)
        after = self.screen_to_world(px, py, width, height)
        cx, cy = self.center
        self.center = (cx + before[0] - after[0], cy + before[1] - after[1])

    def zoom_center(self, factor: float) -> None:
        """Scale the view by `factor` around the window centre (keyboard zoom)."""
        if factor <= 0.0:
            return
        self.view_half_y = min(
            CAMERA_MAX_HALF,
            max(CAMERA_MIN_HALF, self.view_half_y / factor),
        )

    def clamp_to_view(self, point: Point, width: float, height: float) -> Point:
        """Keep a world point inside the *currently visible* rectangle.

        This replaces `clamp_to_grid(point, HALF)`: clamping to the fixed +/-HALF
        square made the tip stick to the old domain while panning/zooming moved
        the real domain elsewhere. Clamping to `visible_rect` follows the camera.
        """
        min_x, min_y, max_x, max_y = self.visible_rect(width, height)
        x = max(min_x, min(max_x, point[0]))
        y = max(min_y, min(max_y, point[1]))
        return x, y

    # -- grid --------------------------------------------------------------

    def grid_step(self, width: float, height: float) -> float:
        """World size of one grid square for the current zoom.

        A fixed 1-unit step would look like an empty screen when zoomed in (lines
        further apart than the viewport) and like a solid block when zoomed out
        (thousands of lines). So the step snaps to the largest human 1/2/5 x 10^k
        value whose squares stay at or below `GRID_MAX_SPACING_PX` on screen.
        Rounding *down* keeps one world unit at 1:1 zoom instead of subdividing.
        """
        pixels_per_unit = max(height, 1.0) / (2.0 * self.view_half_y)
        step = nice_step_at_most(GRID_MAX_SPACING_PX / pixels_per_unit)
        min_x, min_y, max_x, max_y = self.visible_rect(width, height)
        while grid_line_count(min_x, min_y, max_x, max_y, step) > GRID_MAX_LINES:
            step = next_nice_step(step)
        return step
