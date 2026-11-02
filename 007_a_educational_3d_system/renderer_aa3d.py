"""GPU anti-aliased 3D renderer: screen-space capsules, one fragment distance field.

Same public API as ``renderer3d.GridArrowRenderer`` — ``draw_grid``,
``draw_arrow``, ``draw_line``, ``draw_circle``, ``draw_text``, ``measure_text``,
``close`` — so the scene can switch one call between the two without changing
anything else.

How the 3D version works:

1. every segment becomes a quad whose vertices carry both endpoints, a side
   sign and its half width in **pixels**;
2. the vertex shader projects both endpoints to screen pixels, offsets the quad
   perpendicular to the projected line, and writes the corner back as clip
   coordinates (multiplying by ``w`` so perspective division lands on the
   intended screen position);
3. the fragment shader measures the exact distance from the fragment to the
   segment in pixels and fades over one pixel: ``coverage = clamp(half_width +
   0.5 - distance, 0, 1)``;
4. segments crossing the camera's near plane are clipped in homogeneous clip
   space, so zooming into the scene does not produce projected streaks.

No MSAA, no supersampled framebuffer, no post-process pass. Because widths are
stated in pixels, strokes keep their on-screen weight at any orbit, zoom, window
size or HiDPI scale.
"""

from __future__ import annotations

import font3d
import geometry3d
import geometry_aa3d
import glfw
from camera3d import OrbitCamera
from gl_utils3d import Strokes, build_program
from math3d import to_gl
from OpenGL.GL import (
    GL_BLEND,
    GL_DEPTH_TEST,
    GL_DYNAMIC_DRAW,
    GL_FALSE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    glBlendFunc,
    glDeleteProgram,
    glDisable,
    glEnable,
    glGetUniformLocation,
    glIsEnabled,
    glUniform1f,
    glUniform2f,
    glUniformMatrix4fv,
    glUseProgram,
)
from settings3d import (
    AA_ARROW_WIDTH_PX,
    AA_CIRCLE_WIDTH_PX,
    AA_FEATHER_PX,
    AA_GRID_WIDTH_PX,
    AA_LINE_WIDTH_PX,
    AA_TEXT_WIDTH_PX,
    ARROW_COLOR,
)

VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec3 segment_a;    // fixed first endpoint of the segment
layout(location = 1) in vec3 segment_b;    // fixed second endpoint
layout(location = 2) in vec3 color;
layout(location = 3) in float end;          // 0 = corner sits at A, 1 = at B
layout(location = 4) in float side;         // +1 / -1: which side of the line
layout(location = 5) in float half_width;   // stroke half width in pixels

uniform mat4 mvp;
uniform vec2 viewport;      // framebuffer size in pixels
uniform float feather;      // extra margin (pixels) that holds the AA fade

out vec3 v_color;
noperspective out vec2 v_px;
flat out vec2 v_segment_a;
flat out vec2 v_segment_b;
flat out float v_half_width;

const float NEAR_W = 0.05;  // clip-space near plane used for segment clipping

void main() {
    vec4 ca = mvp * vec4(segment_a, 1.0);
    vec4 cb = mvp * vec4(segment_b, 1.0);

    // Clip the 3D segment against the near plane in homogeneous space. Both
    // endpoints are available here, so every corner of the quad agrees on the
    // clipped segment (no gaps or projected streaks near the camera).
    float da = ca.w - NEAR_W;
    float db = cb.w - NEAR_W;
    if (da < 0.0 && db < 0.0) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);   // entirely behind: culled
        v_color = color;
        v_px = vec2(0.0);
        v_segment_a = vec2(0.0);
        v_segment_b = vec2(0.0);
        v_half_width = 0.0;
        return;
    }
    if (da < 0.0) {
        ca = mix(ca, cb, da / (da - db));
    } else if (db < 0.0) {
        cb = mix(cb, ca, db / (db - da));
    }

    vec2 a_px = (ca.xy / ca.w * 0.5 + 0.5) * viewport;
    vec2 b_px = (cb.xy / cb.w * 0.5 + 0.5) * viewport;

    vec2 delta = b_px - a_px;
    float len = length(delta);
    vec2 dir = len > 1e-6 ? delta / len : vec2(1.0, 0.0);
    vec2 normal = vec2(-dir.y, dir.x);

    // Every corner of this quad carries the same A/B pair, so the fragment
    // distance field is well defined. "end" only chooses the anchor corner.
    vec2 anchor_px = mix(a_px, b_px, end);
    float cap = end * 2.0 - 1.0;   // A extends backwards, B forwards
    vec2 corner = anchor_px + normal * side * (half_width + feather) + dir * cap * feather;

    // Back to clip space. Multiplying the NDC offset by the anchor's w makes
    // perspective division land exactly on corner; the endpoint's depth is
    // good enough for these thin ribbons.
    vec4 anchor_clip = mix(ca, cb, end);
    vec2 ndc = corner / viewport * 2.0 - 1.0;
    gl_Position = vec4(ndc * anchor_clip.w, anchor_clip.z, anchor_clip.w);

    v_color = color;
    v_px = corner;
    v_segment_a = a_px;
    v_segment_b = b_px;
    v_half_width = half_width;
}
"""

FRAGMENT_SHADER = """#version 330 core
in vec3 v_color;
noperspective in vec2 v_px;
flat in vec2 v_segment_a;
flat in vec2 v_segment_b;
flat in float v_half_width;

out vec4 frag_color;

// Exact Euclidean distance from p to segment a-b, in pixels.
float distance_to_segment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ab = b - a;
    float t = clamp(dot(pa, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    return length(pa - ab * t);
}

void main() {
    float distance = distance_to_segment(v_px, v_segment_a, v_segment_b);

    // 50% coverage exactly on the geometric edge, fading out over one pixel.
    float coverage = clamp(v_half_width + 0.5 - distance, 0.0, 1.0);
    if (coverage <= 0.001) {
        discard;
    }
    frag_color = vec4(v_color, coverage);
}
"""

SCREEN_VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec3 segment_a;    // pixel-space first endpoint
layout(location = 1) in vec3 segment_b;    // pixel-space second endpoint
layout(location = 2) in vec3 color;
layout(location = 3) in float end;          // 0 = corner sits at A, 1 = at B
layout(location = 4) in float side;         // +1 / -1: which side of the line
layout(location = 5) in float half_width;   // stroke half width in pixels

uniform vec2 screen_size;   // window size, pixels/points, origin top-left
uniform float feather;      // extra margin (pixels) that holds the AA fade

out vec3 v_color;
noperspective out vec2 v_px;
flat out vec2 v_segment_a;
flat out vec2 v_segment_b;
flat out float v_half_width;

void main() {
    vec2 a_px = segment_a.xy;
    vec2 b_px = segment_b.xy;

    vec2 delta = b_px - a_px;
    float len = length(delta);
    vec2 dir = len > 1e-6 ? delta / len : vec2(1.0, 0.0);
    vec2 normal = vec2(-dir.y, dir.x);

    vec2 anchor_px = mix(a_px, b_px, end);
    float cap = end * 2.0 - 1.0;   // A extends backwards, B forwards
    vec2 corner = anchor_px + normal * side * (half_width + feather) + dir * cap * feather;

    // Window pixels (top-left origin, y down) -> OpenGL NDC.
    vec2 ndc = vec2(
        corner.x / screen_size.x * 2.0 - 1.0,
        1.0 - corner.y / screen_size.y * 2.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);

    v_color = color;
    v_px = corner;
    v_segment_a = a_px;
    v_segment_b = b_px;
    v_half_width = half_width;
}
"""


class AAGridArrowRenderer:
    """GPU anti-aliased 3D renderer; the window supplies the size, the camera the view."""

    def __init__(self, window, camera: OrbitCamera) -> None:
        self.window = window
        self.camera = camera
        self.program = build_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.mvp_location = glGetUniformLocation(self.program, "mvp")
        self.viewport_location = glGetUniformLocation(self.program, "viewport")
        self.feather_location = glGetUniformLocation(self.program, "feather")

        self.grid = Strokes()
        self.strokes = Strokes()
        self.screen_strokes = Strokes()
        self.grid_key: tuple | None = None

        # Screen-space text has its own tiny vertex shader: same pixel-distance
        # fragment shader, but pixel coordinates instead of a 3D MVP.
        self.screen_program = build_program(SCREEN_VERTEX_SHADER, FRAGMENT_SHADER)
        self.screen_size_location = glGetUniformLocation(self.screen_program, "screen_size")
        self.screen_feather_location = glGetUniformLocation(self.screen_program, "feather")

        # Coverage arrives as alpha, so the stencil-free way to apply it is blending.
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(self.program)

    # -- internals ---------------------------------------------------------
    def _framebuffer(self) -> tuple[int, int]:
        return glfw.get_framebuffer_size(self.window)

    def _bind_camera(self) -> None:
        glUseProgram(self.program)
        width, height = self._framebuffer()
        glUniformMatrix4fv(
            self.mvp_location,
            1,
            GL_FALSE,
            to_gl(self.camera.mvp(width / max(height, 1))),
        )
        glUniform2f(self.viewport_location, float(width), float(height))
        glUniform1f(self.feather_location, AA_FEATHER_PX)

    def _upload_grid_if_needed(self) -> None:
        """Rebuild only when the viewport, zoom, pivot or adaptive step changed."""
        size = self._framebuffer()
        step = self.camera.grid_step(size[1])
        extent = self.camera.grid_extent()
        key = (size, step, extent, float(self.camera.target[0]), float(self.camera.target[1]))
        if key == self.grid_key:
            return
        self.grid.upload(
            geometry_aa3d.stroke_vertices(
                geometry3d.grid_vertices(
                    float(self.camera.target[0]),
                    float(self.camera.target[1]),
                    extent,
                    step,
                ),
                AA_GRID_WIDTH_PX * 0.5,
            ),
            GL_DYNAMIC_DRAW,
        )
        self.grid_key = key

    def _draw_strokes(self, vertices, half_width_px) -> None:
        self.strokes.upload(
            geometry_aa3d.stroke_vertices(vertices, half_width_px),
            GL_DYNAMIC_DRAW,
        )
        self._bind_camera()
        self.strokes.draw()

    # -- the renderer API shared with renderer3d.GridArrowRenderer ---------
    def draw_grid(self) -> None:
        self._upload_grid_if_needed()
        self._bind_camera()
        self.grid.draw()

    def draw_arrow(
        self,
        start: geometry3d.Point3,
        tip: geometry3d.Point3,
        color: geometry3d.Color = ARROW_COLOR,
    ) -> None:
        """Draw an arrow from `start` to `tip`; `color` is an (r, g, b) tuple."""
        self._draw_strokes(
            geometry3d.arrow_vertices(
                start,
                tip,
                view_dir=self.camera.forward(),
                color=color,
            ),
            AA_ARROW_WIDTH_PX * 0.5,
        )

    def draw_line(
        self,
        start: geometry3d.Point3,
        tip: geometry3d.Point3,
        color: geometry3d.Color = ARROW_COLOR,
        thickness: float = 0.0,
    ) -> None:
        """Draw an anti-aliased segment with rounded caps.

        ``thickness`` is in world units, exactly like the 2D API; ``0`` falls
        back to ``AA_LINE_WIDTH_PX``. The world thickness is converted to a
        pixel half width at the segment midpoint, so a marker keeps the same
        apparent weight as the rest of a 3D scene.
        """
        vertices = geometry3d.line_vertices(start, tip, color=color)
        if thickness <= 0.0:
            half_width = AA_LINE_WIDTH_PX * 0.5
        else:
            half_width = geometry_aa3d.world_thickness_to_pixels(
                vertices,
                thickness,
                self.camera,
                self._framebuffer()[1],
            )
        self._draw_strokes(vertices, half_width)

    def draw_circle(
        self,
        center: geometry3d.Point3,
        radius: float,
        color: geometry3d.Color = ARROW_COLOR,
        axis: geometry3d.Vector3 = (0.0, 0.0, 1.0),
        segments: int = 64,
    ) -> None:
        """Draw a circle in the plane whose normal is ``axis``.

        The outline is the usual ring of segments, but each one is a capsule, so
        the joints are round and the shader fades the rim. ``axis`` is the one
        piece of information a 3D circle needs that a 2D one did not.
        """
        self._draw_strokes(
            geometry3d.circle_vertices(
                center,
                radius,
                color=color,
                axis=axis,
                segments=segments,
            ),
            AA_CIRCLE_WIDTH_PX * 0.5,
        )

    def draw_text(
        self,
        anchor: geometry3d.Point3,
        text: str,
        size: float = 0.30,
        color: geometry3d.Color = ARROW_COLOR,
        *,
        right: geometry3d.Vector3 = (1.0, 0.0, 0.0),
        up: geometry3d.Vector3 = (0.0, 1.0, 0.0),
        tracking: float = font3d.TRACKING,
        width_px: float = AA_TEXT_WIDTH_PX,
    ) -> float:
        """Draw anti-aliased `text` starting at `anchor` (left end of the baseline).

        ``right`` and ``up`` orient the text plane; the defaults are the xy plane
        (the 2D behaviour). ``size`` is the cap height in world units, so text
        scales with the scene. Returns the advance width in world units, handy
        for centring a label or anchoring something after the string.
        """
        vertices = geometry_aa3d.text_vertices(
            anchor,
            text,
            size,
            color=color,
            right=right,
            up=up,
            tracking=tracking,
        )
        if vertices.shape[0]:
            self._draw_strokes(vertices, width_px * 0.5)
        return geometry_aa3d.text_width(text, size, tracking)

    def measure_text(
        self,
        text: str,
        size: float = 0.30,
        tracking: float = font3d.TRACKING,
    ) -> float:
        """Advance width of world-space `text` in world units, for layout."""
        return geometry_aa3d.text_width(text, size, tracking)

    def draw_screen_text(
        self,
        x: float,
        y: float,
        text: str,
        size_px: float = 16.0,
        color: geometry3d.Color = ARROW_COLOR,
        *,
        tracking: float = font3d.TRACKING,
        width_px: float = AA_TEXT_WIDTH_PX,
    ) -> float:
        """Draw anti-aliased text in **screen space**, on top of the 3D scene.

        ``(x, y)`` is the left end of the baseline in window pixels, with the
        origin at the **top-left** and y growing down (the same convention as
        ``glfw.get_cursor_pos``). ``size_px`` is the cap height in pixels, so the
        text keeps a fixed on-screen size regardless of the camera. ``color`` is
        the usual ``(r, g, b)`` tuple; unlike world-space text it is not
        foreshortened, rotated or depth-tested.

        Returns the advance width in pixels, so a caller can right-align or
        anchor something after the string. Text is AA-only, like ``draw_text``.
        """
        vertices = geometry_aa3d.text_vertices(
            (x, y, 0.0),
            text,
            size_px,
            color=color,
            right=(1.0, 0.0, 0.0),
            up=(0.0, -1.0, 0.0),  # screen y grows down
            tracking=tracking,
        )
        if vertices.shape[0]:
            self.screen_strokes.upload(
                geometry_aa3d.stroke_vertices(vertices, width_px * 0.5),
                GL_DYNAMIC_DRAW,
            )
            glUseProgram(self.screen_program)
            width, height = glfw.get_window_size(self.window)
            glUniform2f(self.screen_size_location, float(width), float(height))
            glUniform1f(self.screen_feather_location, AA_FEATHER_PX)

            depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
            glDisable(GL_DEPTH_TEST)  # HUD always sits on top of the scene
            self.screen_strokes.draw()
            if depth_was_enabled:
                glEnable(GL_DEPTH_TEST)

        return self.measure_screen_text(text, size_px, tracking)

    def measure_screen_text(
        self,
        text: str,
        size_px: float = 16.0,
        tracking: float = font3d.TRACKING,
    ) -> float:
        """Advance width of screen-space `text` in window pixels."""
        return geometry_aa3d.text_width(text, size_px, tracking)

    def close(self) -> None:
        self.screen_strokes.destroy()
        self.strokes.destroy()
        self.grid.destroy()
        glDeleteProgram(self.screen_program)
        glDeleteProgram(self.program)

    def __enter__(self) -> AAGridArrowRenderer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
