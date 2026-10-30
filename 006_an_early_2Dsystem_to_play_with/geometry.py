"""Pure 2D geometry and the screen <-> world mapping.

No GLFW or OpenGL imports here, so the math can be tested or reused on its own.
The grid is a batch of GL_LINES vertices: ``(x, y, z, r, g, b)`` per vertex.
"""

from __future__ import annotations

import math

import numpy as np
from settings import (
    ARROW_COLOR,
    ARROW_HEAD_ANGLE,
    ARROW_HEAD_LENGTH,
    AXIS_COLOR,
    GRID_COLOR,
    HALF,
    STEP,
    TAIL_MARKER,
)

Point = tuple[float, float]
Color = tuple[float, float, float]


def grid_vertices(
    half: float = HALF,
    step: float = STEP,
    grid_color: Color = GRID_COLOR,
    axis_color: Color = AXIS_COLOR,
) -> np.ndarray:
    """11 horizontal + 11 vertical lines; the two centre lines are the axes."""
    rows: list[tuple[float, ...]] = []
    extent = half * step
    for i in range(-int(half), int(half) + 1):
        offset = i * step
        color = axis_color if i == 0 else grid_color
        rows += [(-extent, offset, 0.0, *color), (extent, offset, 0.0, *color)]
        rows += [(offset, -extent, 0.0, *color), (offset, extent, 0.0, *color)]
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


def half_extents(width: float, height: float, half: float = HALF) -> tuple[float, float]:
    """World units visible from the centre on x and y — at least `half` on both."""
    aspect = max(width, 1.0) / max(height, 1.0)
    return (half * aspect, half) if aspect >= 1.0 else (half, half / aspect)


def world_to_clip(width: float, height: float, half: float = HALF) -> tuple[float, float]:
    """The 2D scale the vertex shader applies to world coordinates."""
    half_x, half_y = half_extents(width, height, half)
    return 1.0 / half_x, 1.0 / half_y


def screen_to_world(
    px: float,
    py: float,
    width: float,
    height: float,
    half: float = HALF,
) -> Point:
    """GLFW cursor (pixels, y down) -> world (units, y up). Exact inverse of the shader scale."""
    half_x, half_y = half_extents(width, height, half)
    ndc_x = 2.0 * px / max(width, 1.0) - 1.0
    ndc_y = 2.0 * py / max(height, 1.0) - 1.0
    return ndc_x * half_x, -ndc_y * half_y


def clamp_to_grid(point: Point, half: float = HALF) -> Point:
    """Keep a point inside the drawn grid."""
    return (max(-half, min(half, point[0])), max(-half, min(half, point[1])))
