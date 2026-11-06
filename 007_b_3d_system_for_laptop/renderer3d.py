"""Plain 3D renderer: one MVP shader, GL_LINES batches, hard edges.

This is the 3D twin of ``experimental/renderer.py``. It shares the drawing API
with ``renderer_aa3d.py`` so the scene can switch a shape back to aliased lines
by uncommenting one call. The only difference in the geometry is the camera:
every batch is multiplied by ``projection * view`` instead of the 2D camera's
``vec4`` scale/offset.
"""

from __future__ import annotations

import geometry3d
import glfw
import numpy as np
import texture3d
from gl_utils3d import (
    POINT_VERTEX_SHADER,
    POINT_VERTEX_SHADER_RGBA,
    Lines,
    Points,
    PointsRGBA,
    build_program,
    max_point_size,
)
from math3d import to_gl
from OpenGL.GL import (
    GL_BLEND,
    GL_DEPTH_WRITEMASK,
    GL_DYNAMIC_DRAW,
    GL_FALSE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PROGRAM_POINT_SIZE,
    GL_SRC_ALPHA,
    GL_TRUE,
    glBlendFunc,
    glDeleteProgram,
    glDepthMask,
    glDisable,
    glEnable,
    glGetBooleanv,
    glGetUniformLocation,
    glIsEnabled,
    glUniform1f,
    glUniform1i,
    glUniform3f,
    glUniformMatrix4fv,
    glUseProgram,
)
from settings3d import ARROW_COLOR, POINT_MAX_PX, POINT_MIN_PX

VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

uniform mat4 mvp;

out vec3 v_color;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    v_color = color;
}
"""

FRAGMENT_SHADER = """#version 330 core
in vec3 v_color;
out vec4 frag_color;

void main() {
    frag_color = vec4(v_color, 1.0);
}
"""

# Particles: no AA here, so the sprite is a plain square, one fragment each.
POINT_FRAGMENT_SHADER = """#version 330 core
in vec3 v_color;
out vec4 frag_color;

void main() {
    frag_color = vec4(v_color, 1.0);
}
"""

# The same square with the coverage channel: the particle's own alpha (e.g. a
# life-time fade) is written straight through, one assignment in a shader that
# already runs per fragment. Used only by draw_points_rgba.
POINT_FRAGMENT_SHADER_RGBA = """#version 330 core
in vec4 v_color;
out vec4 frag_color;

void main() {
    frag_color = v_color;
}
"""


class GridArrowRenderer:
    """Owns the GL objects; the window supplies the size, the camera the view."""

    def __init__(self, window, camera) -> None:
        self.window = window
        self.camera = camera
        self.program = build_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.mvp_location = glGetUniformLocation(self.program, "mvp")

        self.lines = Lines()
        self.grid = Lines()
        self.grid_key: tuple | None = None
        glUseProgram(self.program)

        # Particles. GL_PROGRAM_POINT_SIZE lets the vertex shader write
        # gl_PointSize, which is what carries the per-point radius; the driver's
        # own limit is folded into the CPU cap so a sprite never silently stops
        # growing. Both renderers enable the same switch (it is global state).
        self.points = Points()
        self.point_program = build_program(POINT_VERTEX_SHADER, POINT_FRAGMENT_SHADER)
        self.point_mvp_location = glGetUniformLocation(self.point_program, "mvp")
        self.point_eye_location = glGetUniformLocation(self.point_program, "eye")
        self.point_forward_location = glGetUniformLocation(self.point_program, "forward")
        self.point_scale_location = glGetUniformLocation(self.point_program, "pixel_scale")
        self.point_min_location = glGetUniformLocation(self.point_program, "min_radius_px")
        self.point_max_location = glGetUniformLocation(self.point_program, "max_radius_px")
        self.point_flat_depth_location = glGetUniformLocation(self.point_program, "flat_depth")
        self.point_feather_location = glGetUniformLocation(self.point_program, "feather")
        self.max_point_px = min(POINT_MAX_PX, max_point_size())
        glEnable(GL_PROGRAM_POINT_SIZE)

        # The alpha-carrying particle twin: the same vertex shader with a vec4
        # colour, its own 8-float VAO (PointsRGBA) and its own program, so the
        # opaque path above keeps its exact buffers, attributes and shader.
        self.points_rgba = PointsRGBA()
        self.point_program_rgba = build_program(
            POINT_VERTEX_SHADER_RGBA, POINT_FRAGMENT_SHADER_RGBA
        )
        self.point_rgba_mvp_location = glGetUniformLocation(self.point_program_rgba, "mvp")
        self.point_rgba_eye_location = glGetUniformLocation(self.point_program_rgba, "eye")
        self.point_rgba_forward_location = glGetUniformLocation(self.point_program_rgba, "forward")
        self.point_rgba_scale_location = glGetUniformLocation(
            self.point_program_rgba, "pixel_scale"
        )
        self.point_rgba_min_location = glGetUniformLocation(
            self.point_program_rgba, "min_radius_px"
        )
        self.point_rgba_max_location = glGetUniformLocation(
            self.point_program_rgba, "max_radius_px"
        )
        self.point_rgba_flat_depth_location = glGetUniformLocation(
            self.point_program_rgba, "flat_depth"
        )
        self.point_rgba_feather_location = glGetUniformLocation(
            self.point_program_rgba, "feather"
        )

        # Pictures: one textured-quad program plus the texture cache that keeps
        # a source from being uploaded again every frame.
        self.images = texture3d.ImageBatch()
        self.textures = texture3d.TextureStore()

    # -- internals ---------------------------------------------------------
    def _bind_camera(self) -> None:
        glUseProgram(self.program)
        width, height = glfw.get_framebuffer_size(self.window)
        glUniformMatrix4fv(
            self.mvp_location,
            1,
            GL_FALSE,
            to_gl(self.camera.mvp(width / max(height, 1))),
        )

    def _upload_grid_if_needed(self) -> None:
        """Rebuild when the viewport, zoom, pivot or adaptive step changed."""
        size = glfw.get_framebuffer_size(self.window)
        step = self.camera.grid_step(size[1])
        extent = self.camera.grid_extent()
        key = (size, step, extent, float(self.camera.target[0]), float(self.camera.target[1]))
        if key == self.grid_key:
            return
        self.grid.upload(
            geometry3d.grid_vertices(
                float(self.camera.target[0]),
                float(self.camera.target[1]),
                extent,
                step,
            ),
            GL_DYNAMIC_DRAW,
        )
        self.grid_key = key

    def _draw(self, vertices) -> None:
        self.lines.upload(vertices, GL_DYNAMIC_DRAW)
        self._bind_camera()
        self.lines.draw()

    def _bind_points(self) -> None:
        glUseProgram(self.point_program)
        width, height = glfw.get_framebuffer_size(self.window)
        glUniformMatrix4fv(
            self.point_mvp_location,
            1,
            GL_FALSE,
            to_gl(self.camera.mvp(width / max(height, 1))),
        )
        eye = self.camera.eye()
        forward = self.camera.forward()
        glUniform3f(self.point_eye_location, float(eye[0]), float(eye[1]), float(eye[2]))
        glUniform3f(
            self.point_forward_location, float(forward[0]), float(forward[1]), float(forward[2])
        )
        glUniform1f(self.point_scale_location, self.camera.pixel_scale(height))
        glUniform1f(self.point_min_location, POINT_MIN_PX)
        glUniform1f(self.point_max_location, 0.5 * self.max_point_px)
        glUniform1i(self.point_flat_depth_location, 0)  # depth-correct, like GL_LINES
        glUniform1f(self.point_feather_location, 0.0)

    # -- the alpha-carrying particle internals (new; the opaque path above is
    #    untouched, so it keeps its exact buffers, attributes and shader) ----
    def _bind_points_rgba(self) -> None:
        """Bind the alpha-carrying point program; ``_bind_points`` is untouched."""
        glUseProgram(self.point_program_rgba)
        width, height = glfw.get_framebuffer_size(self.window)
        glUniformMatrix4fv(
            self.point_rgba_mvp_location,
            1,
            GL_FALSE,
            to_gl(self.camera.mvp(width / max(height, 1))),
        )
        eye = self.camera.eye()
        forward = self.camera.forward()
        glUniform3f(
            self.point_rgba_eye_location, float(eye[0]), float(eye[1]), float(eye[2])
        )
        glUniform3f(
            self.point_rgba_forward_location,
            float(forward[0]),
            float(forward[1]),
            float(forward[2]),
        )
        glUniform1f(self.point_rgba_scale_location, self.camera.pixel_scale(height))
        glUniform1f(self.point_rgba_min_location, POINT_MIN_PX)
        glUniform1f(self.point_rgba_max_location, 0.5 * self.max_point_px)
        glUniform1i(self.point_rgba_flat_depth_location, 0)  # depth-correct, like GL_LINES
        glUniform1f(self.point_rgba_feather_location, 0.0)

    def _draw_points_rgba(self, batch: np.ndarray, depth_write: bool = True) -> None:
        """Upload + draw one 8-float batch, blended over the scene.

        The blend switch and the depth mask are saved and restored the same way
        ``texture3d.ImageBatch.draw`` does it, so nothing outside this call
        inherits a changed state and the opaque path keeps its exact cost.
        """
        self.points_rgba.upload(batch)
        self._bind_points_rgba()

        blend_was_enabled = bool(glIsEnabled(GL_BLEND))
        if not blend_was_enabled:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        depth_mask = bool(np.asarray(glGetBooleanv(GL_DEPTH_WRITEMASK)).ravel()[0])
        if not depth_write and depth_mask:
            glDepthMask(GL_FALSE)

        self.points_rgba.draw()

        if not depth_write and depth_mask:
            glDepthMask(GL_TRUE)
        if not blend_was_enabled:
            glDisable(GL_BLEND)

    # -- the renderer API shared with renderer_aa3d.AAGridArrowRenderer ----
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
        self._draw(
            geometry3d.arrow_vertices(
                start,
                tip,
                view_dir=self.camera.forward(),
                color=color,
            )
        )

    def draw_line(
        self,
        start: geometry3d.Point3,
        tip: geometry3d.Point3,
        color: geometry3d.Color = ARROW_COLOR,
        thickness: float = 0.0,
    ) -> None:
        """Draw a plain 3D segment.

        ``thickness`` is in world units; ``0`` keeps the thin one-pixel
        GL_LINES path. A positive thickness becomes a camera-facing ribbon,
        because GL_LINES itself cannot change width.
        """
        if thickness > 0.0:
            self.lines.upload(
                geometry3d.thick_line_vertices(
                    start,
                    tip,
                    thickness,
                    view_dir=self.camera.forward(),
                    color=color,
                ),
                GL_DYNAMIC_DRAW,
            )
            self._bind_camera()
            self.lines.draw_triangles()
            return
        self._draw(geometry3d.line_vertices(start, tip, color=color))

    def draw_circle(
        self,
        center: geometry3d.Point3,
        radius: float,
        color: geometry3d.Color = ARROW_COLOR,
        axis: geometry3d.Vector3 = (0.0, 0.0, 1.0),
        segments: int = 64,
    ) -> None:
        """Draw a circle outline in the plane whose normal is ``axis``."""
        self._draw(
            geometry3d.circle_vertices(
                center,
                radius,
                color=color,
                axis=axis,
                segments=segments,
            )
        )

    def draw_points(
        self,
        points: geometry3d.Points3,
        size: float | np.ndarray,
        color: geometry3d.Color | np.ndarray = ARROW_COLOR,
    ) -> None:
        """Draw a field of particles in one draw call, one vertex per particle.

        ``points`` is an ``(N, 3)`` array-like (or one ``(x, y, z)`` triple),
        ``size`` is the particle **radius in world units** — a scalar or one
        value per point — and ``color`` is one ``(r, g, b)`` or ``(N, 3)``.
        GL expands each sprite itself, so no geometry is tessellated here; the
        plain renderer draws them as hard squares and depth-tests them like any
        other aliased draw. The vertex shader converts the world-space radius to
        pixels per point, clamps it to the driver's sprite limit, and the
        pipeline clips anything behind the eye.

        For per-particle transparency (a life-time fade) use ``draw_points_rgba``,
        which is a separate method on a separate 8-float buffer so this opaque
        path stays exactly as fast as it was.
        """
        batch = geometry3d.point_vertices(points, size, color=color)
        if not batch.shape[0]:
            return
        self.points.upload(batch)
        self._bind_points()
        self.points.draw()

    def draw_point(
        self,
        center: geometry3d.Point3,
        size: float,
        color: geometry3d.Color = ARROW_COLOR,
    ) -> None:
        """Draw one particle; ``size`` is the radius in world units."""
        self.draw_points(center, size, color)

    def draw_points_sorted(
        self,
        points: geometry3d.Points3,
        size: float | np.ndarray,
        color: geometry3d.Color | np.ndarray = ARROW_COLOR,
        *,
        order: np.ndarray | None = None,
    ) -> None:
        """Draw particles back-to-front, so the nearest one is painted last.

        Same batch, same upload, same single draw call as ``draw_points``, plus
        one view-depth sort (``OrbitCamera.depth_order``) and one gather of the
        packed vertices. Pass ``order`` to reuse a permutation computed earlier
        — e.g. recompute only when the camera moves.

        The plain renderer already depth-tests its points, so for opaque
        particles this changes nothing visually: sorting here exists so the call
        site works with either renderer and stays correct if you turn blending
        on (transparent particles need painter's order, since the depth buffer
        cannot resolve them). Transparent batches use
        ``draw_points_rgba_sorted``, the same permutation on the alpha path.
        """
        batch = geometry3d.point_vertices(points, size, color=color)
        if not batch.shape[0]:
            return
        if order is None:
            order = self.camera.depth_order(batch[:, :3])
        self.points.upload(batch[order])
        self._bind_points()
        self.points.draw()

    # -- particles with alpha: separate buffer, program and methods ---------
    def draw_points_rgba(
        self,
        points: geometry3d.Points3,
        size: float | np.ndarray,
        color: geometry3d.PointColor | np.ndarray = ARROW_COLOR,
        *,
        depth_write: bool = True,
    ) -> None:
        """``draw_points`` with a coverage channel: a per-particle life-time fade.

        Same one-vertex-per-particle batch, except ``color`` may be one
        ``(r, g, b, a)`` or an ``(N, 4)`` array (``(r, g, b)`` is still accepted
        and packs ``a = 1.0``). ``a`` is the particle's coverage and the fragment
        shader writes it straight through, so ``0`` makes the sprite invisible and
        ``1`` is the old opaque square. Driving it towards ``0`` over a particle's
        life is the whole life-time effect.

        This is a separate method on a separate 8-float VAO and program, so the
        opaque ``draw_points`` path keeps its exact old cost. Here the batch is
        one extra float column (32-byte aligned stride), the draw is still one
        upload and one draw call, and blending is switched on around that call
        only (then restored). Fading therefore costs no extra pass and no extra
        draw call.

        ``depth_write`` defaults to ``True``: a nearly invisible particle then
        still writes depth like an opaque one. Pass ``depth_write=False`` to
        composite the batch like glass (the mask is saved and restored), or use
        ``draw_points_rgba_sorted`` when the field has to layer back-to-front.
        """
        batch = geometry3d.point_vertices_rgba(points, size, color=color)
        if not batch.shape[0]:
            return
        self._draw_points_rgba(batch, depth_write)

    def draw_point_rgba(
        self,
        center: geometry3d.Point3,
        size: float,
        color: geometry3d.PointColor = ARROW_COLOR,
        *,
        depth_write: bool = True,
    ) -> None:
        """Draw one particle with coverage; ``size`` is the radius in world units.

        ``color`` may be ``(r, g, b)`` or ``(r, g, b, a)`` (see
        ``draw_points_rgba``).
        """
        self.draw_points_rgba(center, size, color, depth_write=depth_write)

    def draw_points_rgba_sorted(
        self,
        points: geometry3d.Points3,
        size: float | np.ndarray,
        color: geometry3d.PointColor | np.ndarray = ARROW_COLOR,
        *,
        order: np.ndarray | None = None,
        depth_write: bool = True,
    ) -> None:
        """``draw_points_rgba`` painted back-to-front for glass-like layering.

        Same batch, upload and single blended draw call, plus one view-depth sort
        (``OrbitCamera.depth_order``) and one gather of the packed vertices. Pass
        ``order`` to reuse a permutation computed earlier, e.g. only when the
        camera moves — the same contract as ``draw_points_sorted``.
        """
        batch = geometry3d.point_vertices_rgba(points, size, color=color)
        if not batch.shape[0]:
            return
        if order is None:
            order = self.camera.depth_order(batch[:, :3])
        self._draw_points_rgba(batch[order], depth_write)

    def draw_image(
        self,
        anchor: geometry3d.Point3,
        source: texture3d.ImageSource,
        right: geometry3d.Vector3 = (1.0, 0.0, 0.0),
        up: geometry3d.Vector3 = (0.0, 1.0, 0.0),
        width: float = 1.0,
        height: float | None = None,
        *,
        tint: texture3d.Tint = (1.0, 1.0, 1.0, 1.0),
        depth_write: bool | None = None,
    ) -> None:
        """Draw a picture (texture) in an arbitrary 3D plane.

        ``anchor`` is the picture's **bottom-left** corner, ``right`` and ``up``
        are its own axes (only their direction matters), and ``width`` /
        ``height`` are its size in world units along those axes. ``height=None``
        keeps the source's pixel aspect ratio from ``width``.

        ``source`` is a file path, a Pillow image, an ``(H, W, 3)`` /
        ``(H, W, 4)`` NumPy array, or an already-uploaded
        ``texture3d.Texture``. Paths and in-memory images are cached by
        ``texture3d.TextureStore``, so the same source drawn every frame uploads
        once (a different array per frame is the exception — build one
        ``Texture`` and reuse it).

        ``tint`` multiplies the texel: an ``(r, g, b)`` colour tints the picture
        and an RGBA fades it. ``depth_write`` defaults to ``True`` for an opaque
        picture and ``False`` when it has transparency, so blended fragments can
        see what is behind them.
        """
        texture = self.textures.get(source)
        if height is None:
            height = width * texture.height / max(texture.width, 1)
        vertices = geometry3d.image_quad_vertices(anchor, right, up, width, height)
        width_px, height_px = glfw.get_framebuffer_size(self.window)
        self.images.draw(
            vertices,
            texture,
            self.camera.mvp(width_px / max(height_px, 1)),
            tint,
            depth_write=depth_write,
        )

    def close(self) -> None:
        self.textures.close()
        self.images.close()
        self.points_rgba.destroy()
        self.points.destroy()
        self.lines.destroy()
        self.grid.destroy()
        glDeleteProgram(self.point_program_rgba)
        glDeleteProgram(self.point_program)
        glDeleteProgram(self.program)

    def __enter__(self) -> GridArrowRenderer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
