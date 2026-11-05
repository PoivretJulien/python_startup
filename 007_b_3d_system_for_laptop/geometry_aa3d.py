"""Screen-space stroke geometry for the 3D anti-aliased renderer.

The 2D AA renderer expands a segment into a quad in *world* space and measures
the distance to the segment with a world-space distance field. In 3D that would
make stroke width depend on perspective (and would still alias once the camera
tilts), so this version expands each segment in **screen space** instead:

* the vertex shader projects both endpoints of the segment, converts them to
  pixels, offsets the quad corners perpendicular to the projected line, and
  writes the corner back as clip coordinates;
* the fragment shader measures the distance from the fragment to the segment in
  *pixels*, so the coverage ramp is exactly one pixel wide at any depth or
  angle;
* near-plane clipping happens in homogeneous clip space in the vertex shader,
  which also tames the huge projected coordinates of segments crossing the
  camera plane.

Vertex layout (12 float32 per vertex, uploaded by ``gl_utils3d.Strokes``)::

    ax, ay, az | bx, by, bz | r, g, b | end | side | half_width_px
    segment A    segment B    colour    0/1   +/-1   stroke half width

Both segment endpoints are repeated on all six quad vertices, so the
fragment's pixel-space distance field always refers to the same A-B segment.
``end`` says which endpoint the corner is anchored at (0 = A, 1 = B); the
shader derives the corner position instead of storing it separately.

Like ``geometry3d.py`` this module is pure math: no GLFW, no OpenGL, no shaders.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from font3d import TRACKING, advance, outline
from geometry3d import Color, Point3, Vector3

STRIDE_FLOATS = 12  # segment A(3) + segment B(3) + colour(3) + end + side + width

# The six corners of a segment's quad, in triangle order:
#   (A+, A-, B-) and (A+, B-, B+)
_SIDES = np.array((1.0, -1.0, -1.0, 1.0, -1.0, 1.0), dtype=np.float32)
_ENDS = np.array((0.0, 0.0, 1.0, 0.0, 1.0, 1.0), dtype=np.float32)


def stroke_vertices(
    vertices: np.ndarray,
    half_width_px: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Turn a GL_LINES batch into screen-space anti-aliased stroke quads.

    ``vertices`` uses the same ``(x, y, z, r, g, b)`` layout ``geometry3d.py``
    builds for ``GL_LINES``; consecutive pairs are segments. Each segment takes
    the colour of its *start* vertex, so a batch can mix colours (the grid does).

    ``half_width_px`` is in **pixels** — a scalar for a uniform batch, or one
    value per segment. ``world_thickness_to_pixels`` is the bridge for callers
    that want to state a world-space thickness instead.
    """
    batch = np.ascontiguousarray(vertices, dtype=np.float32).reshape(-1, 6)
    count = batch.shape[0] - batch.shape[0] % 2
    if count < 2:
        return np.zeros((0, STRIDE_FLOATS), dtype=np.float32)

    starts = batch[0:count:2, :3]
    ends = batch[1:count:2, :3]
    colors = batch[0:count:2, 3:6]
    segments = starts.shape[0]

    half_width = np.asarray(half_width_px, dtype=np.float32)
    if half_width.ndim == 0:
        half_width = np.full(segments, float(half_width), dtype=np.float32)
    elif half_width.shape[0] != segments:
        raise ValueError(f"half_width_px has {half_width.shape[0]} values for {segments} segments")

    out = np.empty((segments * 6, STRIDE_FLOATS), dtype=np.float32)
    out[:, 0:3] = np.repeat(starts, 6, axis=0)
    out[:, 3:6] = np.repeat(ends, 6, axis=0)
    out[:, 6:9] = np.repeat(colors, 6, axis=0)
    out[:, 9] = np.tile(_ENDS, segments)
    out[:, 10] = np.tile(_SIDES, segments)
    out[:, 11] = np.repeat(half_width, 6)
    return out


def world_thickness_to_pixels(
    vertices: np.ndarray,
    thickness: float,
    camera,
    height_px: float,
) -> np.ndarray:
    """Per-segment pixel half-width for a stroke stated in world units.

    A 3D line has no single pixels-per-world-unit scale: depth changes along it.
    Sampling at each segment midpoint is enough for the short segments the demo
    uses (circle arcs, arrow heads, markers) and keeps ``draw_line``'s 2D
    meaning — ``thickness`` is in world units.
    """
    batch = np.ascontiguousarray(vertices, dtype=np.float32).reshape(-1, 6)
    count = batch.shape[0] - batch.shape[0] % 2
    if count < 2:
        return np.zeros(0, dtype=np.float32)
    midpoints = (batch[0:count:2, :3] + batch[1:count:2, :3]) * 0.5
    half = np.empty(midpoints.shape[0], dtype=np.float32)
    for i, point in enumerate(midpoints):
        half[i] = 0.5 * thickness * camera.pixels_per_world_unit(point, height_px)
    return half


def _normalized(vector: Vector3) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(v))
    return v / length if length > 1e-12 else np.zeros(3)


def text_vertices(
    anchor: Point3,
    text: str,
    size: float,
    *,
    color: Color,
    right: Vector3 = (1.0, 0.0, 0.0),
    up: Vector3 = (0.0, 1.0, 0.0),
    tracking: float = TRACKING,
) -> np.ndarray:
    """Lay out `text` as a GL_LINES batch in an arbitrary 3D plane.

    ``anchor`` is the left end of the baseline; ``right`` and ``up`` are the
    plane's axes (the defaults are the xy plane, exactly like the 2D app). For
    a label lying on the floor, pass ``right=(1, 0, 0), up=(0, 0, -1)``.
    ``size`` is the cap height in world units, so text scales with the scene.
    """
    origin = np.asarray(anchor, dtype=np.float64)
    r = _normalized(right)
    u = _normalized(up)
    rows: list[tuple[float, ...]] = []
    for shape in outline(text, tracking):
        for (x0, y0), (x1, y1) in zip(shape, shape[1:]):
            p0 = origin + (r * x0 + u * y0) * size
            p1 = origin + (r * x1 + u * y1) * size
            rows.append((p0[0], p0[1], p0[2], *color))
            rows.append((p1[0], p1[1], p1[2], *color))
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def text_width(text: str, size: float, tracking: float = TRACKING) -> float:
    """Advance width of `text` in world units, i.e. where the next glyph starts."""
    return advance(text, tracking) * size
