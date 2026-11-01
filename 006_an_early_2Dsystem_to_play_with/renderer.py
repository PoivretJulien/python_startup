"""Grid + arrow renderer: one shader, two line batches, and the world-to-clip scale."""

from __future__ import annotations

import geometry
import glfw
from gl_utils import Lines, build_program
from OpenGL.GL import (
    GL_DYNAMIC_DRAW,
    glDeleteProgram,
    glGetUniformLocation,
    glUniform4f,
    glUseProgram,
)
from settings import ARROW_COLOR

VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

uniform vec4 camera;            // xy = world->clip scale, zw = world->clip offset

out vec3 v_color;

void main() {
    gl_Position = vec4(position.xy * camera.xy + camera.zw, 0.0, 1.0);
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

    def __init__(self, window, camera: geometry.Camera) -> None:
        self.window = window
        self.camera = camera
        self.program = build_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.camera_location = glGetUniformLocation(self.program, "camera")

        self.grid = Lines()
        self.grid_key: tuple | None = None
        self.arrow = Lines()
        glUseProgram(self.program)

    def _bind_camera(self) -> None:
        glUseProgram(self.program)
        width, height = glfw.get_framebuffer_size(self.window)
        glUniform4f(self.camera_location, *self.camera.world_to_clip(width, height))

    def _upload_grid_if_needed(self) -> None:
        """Re-upload the grid when the visible rectangle or the adaptive step changed."""
        size = glfw.get_framebuffer_size(self.window)
        step = self.camera.grid_step(*size)
        key = (size, step, self.camera.center, self.camera.view_half_y)
        if key == self.grid_key:
            return
        min_x, min_y, max_x, max_y = self.camera.visible_rect(*size)
        self.grid.upload(
            geometry.grid_vertices(min_x, min_y, max_x, max_y, step),
            GL_DYNAMIC_DRAW,
        )
        self.grid_key = key

    def draw_grid(self) -> None:
        self._upload_grid_if_needed()
        self._bind_camera()
        self.grid.draw()

    def draw_arrow(
        self,
        start: geometry.Point,
        tip: geometry.Point,
        color: geometry.Color = ARROW_COLOR,
    ) -> None:
        """Draw an arrow from `start` to `tip`; `color` is an (r, g, b) tuple."""
        self.arrow.upload(
            geometry.arrow_vertices(start, tip, color=color),
            GL_DYNAMIC_DRAW,
        )
        self._bind_camera()
        self.arrow.draw()

    def draw_line(
        self,
        start: geometry.Point,
        tip: geometry.Point,
        color: geometry.Color = ARROW_COLOR,
        thickness: float = 0.0,
    ) -> None:
        """Draw a vector without an arrow head (a plain line segment).

        ``thickness`` is in world units; ``0`` keeps the thin GL_LINES path.
        """
        if thickness > 0.0:
            self.arrow.upload(
                geometry.thick_line_vertices(start, tip, thickness, color=color),
                GL_DYNAMIC_DRAW,
            )
            self._bind_camera()
            self.arrow.draw_triangles()
            return

        self.arrow.upload(
            geometry.line_vertices(start, tip, color=color),
            GL_DYNAMIC_DRAW,
        )
        self._bind_camera()
        self.arrow.draw()

    def draw_circle(
        self,
        center: geometry.Point,
        radius: float,
        color: geometry.Color = ARROW_COLOR,
        segments: int = 64,
    ) -> None:
        """Draw a circle outline centered at `center` with `radius` world units."""
        self.arrow.upload(
            geometry.circle_vertices(center, radius, color=color, segments=segments),
            GL_DYNAMIC_DRAW,
        )
        self._bind_camera()
        self.arrow.draw()

    def close(self) -> None:
        self.arrow.destroy()
        self.grid.destroy()
        glDeleteProgram(self.program)

    def __enter__(self) -> GridArrowRenderer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
