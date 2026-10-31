"""Stroke geometry for the GPU anti-aliased renderer.

The plain renderer feeds ``GL_LINES`` one pixel wide and lets the rasteriser do
whatever it wants at the boundary — that is where the stair-steps come from.
Here every segment is instead expanded into a quad ("capsule") that *carries its
own endpoints*, so the fragment shader can measure the exact distance to the
line and fade the last pixel: coverage computed per fragment, entirely on the
GPU, and no multisample buffers involved.

Vertex layout (10 float32 per vertex, see ``gl_utils.Strokes``)::

    x, y | r, g, b | ax, ay | bx, by | half_width
    pos    colour    segment start   segment end   stroke half width (world)

Like ``geometry.py`` this module is pure math: no GLFW, no OpenGL, no shaders.
"""

from __future__ import annotations

import font
import geometry
import numpy as np
from settings import ARROW_COLOR, HALF

STRIDE_FLOATS = 10  # x, y | r, g, b | ax, ay | bx, by | half_width


def stroke_vertices(
    vertices: np.ndarray,
    half_width: float,
    *,
    pad: float = 0.0,
) -> np.ndarray:
    """Turn a GL_LINES batch into anti-aliased stroke quads (two triangles each).

    ``vertices`` is the ``(x, y, z, r, g, b)`` layout that ``geometry.py`` builds
    for ``GL_LINES``; consecutive pairs are read as segments. Each segment takes
    the colour of its *start* vertex, so a batch can mix colours (the grid does:
    grey lines plus the orange axes) exactly like the ``GL_LINES`` original.

    ``half_width`` is in world units. ``pad`` grows the quad past the stroke by
    that many world units so the shader keeps room for its one-pixel fade; pass
    ``2 * world_units_per_pixel(...)`` and no visible edge gets clipped.
    """
    batch = np.ascontiguousarray(vertices, dtype=np.float32).reshape(-1, 6)
    count = batch.shape[0] - batch.shape[0] % 2
    if count < 2:
        return np.zeros((0, STRIDE_FLOATS), dtype=np.float32)

    color = batch[0:count:2, 3:6]
    start = batch[0:count:2, :2]
    end = batch[1:count:2, :2]

    delta = end - start
    length = np.hypot(delta[:, 0], delta[:, 1])
    safe = length > 1e-9
    direction = delta * np.where(safe, 1.0 / np.where(safe, length, 1.0), 0.0)[:, None]
    direction[~safe] = (1.0, 0.0)  # degenerate segment: the shader draws a dot
    normal = np.stack((-direction[:, 1], direction[:, 0]), axis=1)

    half = np.broadcast_to(np.asarray(half_width, dtype=np.float32), length.shape)
    extent = (half + pad)[:, None]
    reach = direction * extent  # capsule end caps, not butt caps
    offset = normal * extent

    near_up = start - reach + offset
    near_down = start - reach - offset
    far_up = end + reach + offset
    far_down = end + reach - offset

    # Two triangles per segment, in the same order as the repeats below.
    positions = np.stack(
        [near_up, near_down, far_down, near_up, far_down, far_up],
        axis=1,
    ).reshape(-1, 2)

    out = np.empty((positions.shape[0], STRIDE_FLOATS), dtype=np.float32)
    out[:, 0:2] = positions
    out[:, 2:5] = np.repeat(color, 6, axis=0)
    out[:, 5:7] = np.repeat(start, 6, axis=0)
    out[:, 7:9] = np.repeat(end, 6, axis=0)
    out[:, 9] = np.repeat(half, 6)
    return out


def text_vertices(
    anchor: geometry.Point,
    text: str,
    size: float,
    *,
    color: geometry.Color = ARROW_COLOR,
    tracking: float = font.TRACKING,
) -> np.ndarray:
    """Lay out `text` as a GL_LINES batch, ready for :func:`stroke_vertices`.

    `anchor` is the left end of the baseline in world units and `size` is the cap
    height, also in world units (so ``size = 0.4`` means digits are 0.4 units
    tall). The font itself (`font.py`) only knows em units; this is where they
    become world space, which is why text scales with the camera the same way the
    arrows do.
    """
    ax, ay = anchor
    rows: list[tuple[float, ...]] = []
    for shape in font.outline(text, tracking):
        for (x0, y0), (x1, y1) in zip(shape, shape[1:]):
            rows.append((ax + x0 * size, ay + y0 * size, 0.0, *color))
            rows.append((ax + x1 * size, ay + y1 * size, 0.0, *color))
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def text_width(text: str, size: float, tracking: float = font.TRACKING) -> float:
    """Advance width of `text` in world units, i.e. where the next glyph starts."""
    return font.advance(text, tracking) * size


def world_units_per_pixel(width: float, height: float, half: float = HALF) -> float:
    """How much world space one framebuffer pixel covers (same on x and y).

    ``geometry.world_to_clip`` already corrects for aspect, so the x and y scales
    agree: a stroke can be sized in pixels and the shader can turn its distance
    field into pixel units with a single scalar.
    """
    _, half_y = geometry.half_extents(width, height, half)
    return 2.0 * half_y / max(height, 1.0)
