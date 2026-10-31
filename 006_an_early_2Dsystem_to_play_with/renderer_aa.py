"""The anti-aliased twin of ``renderer.GridArrowRenderer``.

Same public API — ``draw_grid``, ``draw_arrow``, ``draw_line``, ``draw_circle``,
``draw_text``, ``close`` — so ``main.py`` can swap one for the other without
touching the scene code. What changes is *how* a stroke becomes pixels:

* the plain renderer uploads ``GL_LINES``: one specific pixel-wide rasteriser
  rule, no width control, hard edges (the stair-steps);
* this renderer expands each segment into a quad that carries its own two
  endpoints and half width, and the fragment shader measures the exact distance
  from the fragment to that segment. Distance -> coverage -> alpha, so the last
  pixel of every edge is a real blend rather than a step.

Everything is per-fragment math on the GPU: no ``glEnable(GL_MULTISAMPLE)``, no
supersampled framebuffer, no post-process pass, and the widths are stated in
*pixels* so they survive window resizes and HiDPI framebuffers.
"""

from __future__ import annotations

import font
import geometry
import geometry_aa
import glfw
from gl_utils import Strokes, build_program
from OpenGL.GL import (
    GL_BLEND,
    GL_DYNAMIC_DRAW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    GL_STATIC_DRAW,
    glBlendFunc,
    glDeleteProgram,
    glEnable,
    glGetUniformLocation,
    glUniform2f,
    glUseProgram,
)
from settings import (
    AA_ARROW_WIDTH_PX,
    AA_CIRCLE_WIDTH_PX,
    AA_FEATHER_PX,
    AA_GRID_WIDTH_PX,
    AA_LINE_WIDTH_PX,
    AA_TEXT_WIDTH_PX,
    ARROW_COLOR,
)

VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec2 position;     // quad corner, world units
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 segment_start; // the stroke this quad belongs to
layout(location = 3) in vec2 segment_end;
layout(location = 4) in float half_width;   // world units

uniform vec2 world_to_clip;      // 2D orthographic scale, aspect-corrected

out vec3 v_color;
out vec2 v_position;
out vec2 v_segment_start;
out vec2 v_segment_end;
out float v_half_width;

void main() {
    gl_Position = vec4(position * world_to_clip, 0.0, 1.0);
    v_color = color;
    v_position = position;
    v_segment_start = segment_start;
    v_segment_end = segment_end;
    v_half_width = half_width;
}
"""

FRAGMENT_SHADER = """#version 330 core
in vec3 v_color;
in vec2 v_position;
in vec2 v_segment_start;
in vec2 v_segment_end;
in float v_half_width;

out vec4 frag_color;

// Exact Euclidean distance from p to the segment a-b: a capsule field.
float distance_to_segment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ab = b - a;
    float t = clamp(dot(pa, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    return length(pa - ab * t);
}

void main() {
    // Signed distance to the stroke boundary: negative inside, zero on the edge.
    float boundary = distance_to_segment(v_position, v_segment_start, v_segment_end)
                     - v_half_width;

    // Derivatives turn that world-space field into pixels. Taking the length of
    // the gradient (instead of fwidth's |ddx| + |ddy|) keeps the fade exactly one
    // pixel wide whatever angle the stroke sits at, so diagonals stay as crisp
    // as the axis-aligned grid lines.
    vec2 gradient = vec2(dFdx(boundary), dFdy(boundary));
    float pixel = max(length(gradient), 1e-9);

    // Coverage is linear across that pixel, 50% right on the geometric edge.
    float coverage = clamp(0.5 - boundary / pixel, 0.0, 1.0);
    if (coverage <= 0.001) {
        discard;                 // outside the quad's capsule: no write at all
    }
    frag_color = vec4(v_color, coverage);
}
"""


class AAGridArrowRenderer:
    """GPU anti-aliased renderer; the window supplies the size for the scale."""

    def __init__(self, window) -> None:
        self.window = window
        self.program = build_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.scale_location = glGetUniformLocation(self.program, "world_to_clip")

        # The grid only changes when the framebuffer does, so it gets its own batch.
        self.grid = Strokes()
        self.strokes = Strokes()
        self.grid_size = (0, 0)

        # Coverage arrives as alpha, so the stencil-free way to apply it is blending.
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(self.program)

    # -- internals ---------------------------------------------------------

    def _framebuffer(self) -> tuple[int, int]:
        return glfw.get_framebuffer_size(self.window)

    def _world_units_per_pixel(self) -> float:
        width, height = self._framebuffer()
        return geometry_aa.world_units_per_pixel(width, height)

    def _half_width(self, width_px: float) -> float:
        """Turn a wanted on-screen width in pixels into world units."""
        return width_px * 0.5 * self._world_units_per_pixel()

    def _feather(self) -> float:
        """Quad padding that keeps the AA fade band inside the geometry."""
        return AA_FEATHER_PX * self._world_units_per_pixel()

    def _bind_scale(self) -> None:
        glUseProgram(self.program)
        width, height = self._framebuffer()
        glUniform2f(self.scale_location, *geometry.world_to_clip(width, height))

    def _upload_grid_if_resized(self) -> None:
        size = self._framebuffer()
        if size == self.grid_size:
            return
        self.grid.upload(
            geometry_aa.stroke_vertices(
                geometry.grid_vertices(),
                self._half_width(AA_GRID_WIDTH_PX),
                pad=self._feather(),
            ),
            GL_STATIC_DRAW,
        )
        self.grid_size = size

    def _draw_strokes(self, vertices) -> None:
        self.strokes.upload(vertices, GL_DYNAMIC_DRAW)
        self._bind_scale()
        self.strokes.draw()

    # -- the renderer API shared with renderer.GridArrowRenderer -----------

    def draw_grid(self) -> None:
        self._upload_grid_if_resized()
        self._bind_scale()
        self.grid.draw()

    def draw_arrow(
        self,
        start: geometry.Point,
        tip: geometry.Point,
        color: geometry.Color = ARROW_COLOR,
    ) -> None:
        """Draw an arrow from `start` to `tip`; `color` is an (r, g, b) tuple."""
        self._draw_strokes(
            geometry_aa.stroke_vertices(
                geometry.arrow_vertices(start, tip, color=color),
                self._half_width(AA_ARROW_WIDTH_PX),
                pad=self._feather(),
            )
        )

    def draw_line(
        self,
        start: geometry.Point,
        tip: geometry.Point,
        color: geometry.Color = ARROW_COLOR,
        thickness: float = 0.0,
    ) -> None:
        """Draw a plain line segment, anti-aliased, with rounded caps.

        ``thickness`` is in world units; ``0`` falls back to ``AA_LINE_WIDTH_PX``
        so the segment still has a real, controllable width in pixels.
        """
        half_width = self._half_width(AA_LINE_WIDTH_PX) if thickness <= 0.0 else thickness * 0.5
        self._draw_strokes(
            geometry_aa.stroke_vertices(
                geometry.line_vertices(start, tip, color=color),
                half_width,
                pad=self._feather(),
            )
        )

    def draw_text(
        self,
        anchor: geometry.Point,
        text: str,
        size: float = 0.30,
        color: geometry.Color = ARROW_COLOR,
        tracking: float = font.TRACKING,
        width_px: float = AA_TEXT_WIDTH_PX,
    ) -> float:
        """Draw anti-aliased `text` starting at `anchor` (left end of the baseline).

        `size` is the cap height in world units, so text scales with the scene
        rather than with the window. `tracking` adds extra letter spacing in em
        units, `width_px` is the stroke weight in pixels. Returns the advance
        width in world units — handy for anchoring something after the string
        (``anchor[0] + draw_text(...)``).

        Each glyph is a polyline from `font.py`, so the same capsule pipeline that
        draws the arrows draws the letters: no texture atlas, no extra shader.
        """
        vertices = geometry_aa.text_vertices(anchor, text, size, color=color, tracking=tracking)
        if vertices.shape[0]:
            self._draw_strokes(
                geometry_aa.stroke_vertices(
                    vertices,
                    self._half_width(width_px),
                    pad=self._feather(),
                )
            )
        return geometry_aa.text_width(text, size, tracking)

    def measure_text(
        self,
        text: str,
        size: float = 0.30,
        tracking: float = font.TRACKING,
    ) -> float:
        """Advance width of `text` in world units, for layout before drawing."""
        return geometry_aa.text_width(text, size, tracking)

    def draw_circle(
        self,
        center: geometry.Point,
        radius: float,
        color: geometry.Color = ARROW_COLOR,
        segments: int = 64,
    ) -> None:
        """Draw a circle outline centered at `center` with `radius` world units.

        The outline is the usual ring of segments, but each one is a capsule, so
        the joints are round and the shader fades the rim: no polygonal stair
        steps even at this segment count.
        """
        self._draw_strokes(
            geometry_aa.stroke_vertices(
                geometry.circle_vertices(center, radius, color=color, segments=segments),
                self._half_width(AA_CIRCLE_WIDTH_PX),
                pad=self._feather(),
            )
        )

    def close(self) -> None:
        self.strokes.destroy()
        self.grid.destroy()
        glDeleteProgram(self.program)

    def __enter__(self) -> AAGridArrowRenderer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
