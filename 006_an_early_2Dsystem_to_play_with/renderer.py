"""Grid + arrow renderer: one shader, two line batches, and the world-to-clip scale."""

from __future__ import annotations

import geometry
import glfw
from gl_utils import Lines, build_program
from OpenGL.GL import (
    GL_DYNAMIC_DRAW,
    glDeleteProgram,
    glGetUniformLocation,
    glUniform2f,
    glUseProgram,
)
from settings import ARROW_COLOR

VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

uniform vec2 world_to_clip;      // 2D orthographic scale, aspect-corrected

out vec3 v_color;

void main() {
    gl_Position = vec4(position.xy * world_to_clip, 0.0, 1.0);
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
    """Owns the GL objects; the window supplies the current size for the scale."""

    def __init__(self, window) -> None:
        self.window = window
        self.program = build_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.scale_location = glGetUniformLocation(self.program, "world_to_clip")

        self.grid = Lines()
        self.grid.upload(geometry.grid_vertices())
        self.arrow = Lines()
        glUseProgram(self.program)

    def _bind_scale(self) -> None:
        glUseProgram(self.program)
        width, height = glfw.get_framebuffer_size(self.window)
        glUniform2f(self.scale_location, *geometry.world_to_clip(width, height))

    def draw_grid(self) -> None:
        self._bind_scale()
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
        self._bind_scale()
        self.arrow.draw()

    def close(self) -> None:
        self.arrow.destroy()
        self.grid.destroy()
        glDeleteProgram(self.program)

    def __enter__(self) -> GridArrowRenderer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
