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
from gl_utils3d import Lines, build_program
from math3d import to_gl
from OpenGL.GL import (
    GL_DYNAMIC_DRAW,
    GL_FALSE,
    glDeleteProgram,
    glGetUniformLocation,
    glUniformMatrix4fv,
    glUseProgram,
)
from settings3d import ARROW_COLOR

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

    def close(self) -> None:
        self.lines.destroy()
        self.grid.destroy()
        glDeleteProgram(self.program)

    def __enter__(self) -> GridArrowRenderer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
