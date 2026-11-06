"""OpenGL helpers for the 3D experiment: program builder and the batch VAOs.

Three batch shapes cover the drawing API plus one for pictures: ``Lines``
(``GL_LINES`` / ``GL_TRIANGLES`` for strokes and ribbons), ``Quads`` (textured
``GL_TRIANGLES``: ``x, y, z, u, v``), ``Strokes`` (screen-space AA quads) and
``Points`` (``GL_POINTS`` particles, one vertex each).
"""

from __future__ import annotations

import ctypes

import numpy as np
from OpenGL.constant import Constant
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COMPILE_STATUS,
    GL_DYNAMIC_DRAW,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_LINES,
    GL_LINK_STATUS,
    GL_POINT_SIZE_RANGE,
    GL_POINTS,
    GL_STATIC_DRAW,
    GL_STREAM_DRAW,
    GL_TRIANGLES,
    GL_VERTEX_SHADER,
    glAttachShader,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteBuffers,
    glDeleteShader,
    glDeleteVertexArrays,
    glDrawArrays,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glGetFloatv,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glLinkProgram,
    glShaderSource,
    glVertexAttribPointer,
)

# A GL enum argument: a plain ``int`` or one of PyOpenGL's ``Constant`` objects
# (which *are* ints at runtime, but are not declared as such).
GLenum = int | Constant


# Shared vertex shader for particles: both renderers use it, and only the
# fragment shader (square vs feathered disc) and the depth rule differ.
POINT_VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in float radius_world;   // particle radius, world units

uniform mat4 mvp;
uniform vec3 eye;           // camera position, world space
uniform vec3 forward;       // eye -> target, unit
uniform float pixel_scale;  // framebuffer height / (2 tan(fov/2))
uniform float min_radius_px;
uniform float max_radius_px;
uniform bool flat_depth;    // AA: every layer sits at NDC z = 0, like the strokes
uniform float feather;      // extra sprite margin that holds the AA fade

out vec3 v_color;
out float v_radius_px;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    if (flat_depth) {
        gl_Position.z = 0.0;
    }

    // The perspective divide happens here, not on the CPU: the caller states a
    // world-space radius, which is the only unit that makes sense in a 3D scene.
    // Points behind the eye still compute a finite size (the max clamp), but the
    // pipeline clips them (w <= 0) before they can be rasterised, so a whole
    // batch needs no filter pass on the Python side.
    float depth = max(dot(position - eye, forward), 1e-6);
    float radius_px = clamp(radius_world * pixel_scale / depth, min_radius_px, max_radius_px);

    // Program-controlled point size: the sprite's diameter in pixels. This is
    // why a particle batch needs no geometry expansion on the CPU at all.
    gl_PointSize = 2.0 * (radius_px + feather);

    v_color = color;
    v_radius_px = radius_px;
}
"""


def max_point_size() -> float:
    """Largest point sprite the driver supports, in pixels (GL guarantees >= 1)."""
    values = glGetFloatv(GL_POINT_SIZE_RANGE)
    return float(values[1]) if len(values) > 1 else 1.0


def _compile_shader(source: str, kind: GLenum) -> int:
    shader = glCreateShader(kind)
    if not shader:
        raise RuntimeError(f"glCreateShader failed for kind {kind}")
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader).decode()
        numbered = "\n".join(f"{i:>3} | {line}" for i, line in enumerate(source.splitlines(), 1))
        raise RuntimeError(f"shader compile failed:\n{log}\n{numbered}")
    return shader


def build_program(vertex_source: str, fragment_source: str) -> int:
    """Compile + link one vertex/fragment pair, with line-numbered errors."""
    program = glCreateProgram()
    if not program:
        raise RuntimeError("glCreateProgram failed")
    stages = [
        _compile_shader(vertex_source, GL_VERTEX_SHADER),
        _compile_shader(fragment_source, GL_FRAGMENT_SHADER),
    ]
    for stage in stages:
        glAttachShader(program, stage)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise RuntimeError(f"link failed:\n{glGetProgramInfoLog(program).decode()}")
    for stage in stages:
        glDeleteShader(stage)
    return program


class Lines:
    """A VAO/VBO for one GL_LINES (or GL_TRIANGLES) batch: ``x, y, z, r, g, b``."""

    STRIDE = 6 * 4  # xyz + rgb, float32

    def __init__(self) -> None:
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        if not self.vao or not self.vbo:
            raise RuntimeError("could not create the line buffers")
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.STRIDE, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, self.STRIDE, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)
        self.count = 0

    def upload(self, vertices: np.ndarray, usage: GLenum = GL_STATIC_DRAW) -> None:
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, usage)
        self.count = int(vertices.shape[0])

    def draw(self) -> None:
        if not self.count:
            return
        glBindVertexArray(self.vao)
        glDrawArrays(GL_LINES, 0, self.count)

    def draw_triangles(self) -> None:
        """Draw the uploaded vertices as GL_TRIANGLES (used by thick plain lines)."""
        if not self.count:
            return
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.count)

    def destroy(self) -> None:
        glDeleteBuffers(1, [self.vbo])
        glDeleteVertexArrays(1, [self.vao])


class Quads:
    """A VAO/VBO for one textured GL_TRIANGLES batch: ``x, y, z, u, v``.

    Used by ``texture3d.ImageBatch`` for pictures: six vertices (two triangles)
    per draw, so the buffer is uploaded per picture rather than accumulated.
    """

    STRIDE = 5 * 4  # xyz + uv, float32

    def __init__(self) -> None:
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        if not self.vao or not self.vbo:
            raise RuntimeError("could not create the quad buffers")
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.STRIDE, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.STRIDE, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)
        self.count = 0

    def upload(self, vertices: np.ndarray, usage: GLenum = GL_DYNAMIC_DRAW) -> None:
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, usage)
        self.count = int(vertices.shape[0])

    def draw(self) -> None:
        if not self.count:
            return
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.count)

    def destroy(self) -> None:
        glDeleteBuffers(1, [self.vbo])
        glDeleteVertexArrays(1, [self.vao])


class Strokes:
    """A VAO/VBO for one screen-space stroke batch, drawn as GL_TRIANGLES.

    One vertex is 12 float32 (see ``geometry_aa3d``)::

        ax, ay, az | bx, by, bz | r, g, b | end | side | half_width_px
    """

    STRIDE = 12 * 4

    ATTRIBUTES = (
        (0, 3, 0),  # segment endpoint A (world), repeated on all six corners
        (1, 3, 3),  # segment endpoint B (world)
        (2, 3, 6),  # colour
        (3, 1, 9),  # which endpoint this corner is anchored at (0 = A, 1 = B)
        (4, 1, 10),  # side of the stroke (+/-1)
        (5, 1, 11),  # stroke half width in pixels
    )

    def __init__(self) -> None:
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        if not self.vao or not self.vbo:
            raise RuntimeError("could not create the stroke buffers")
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        for location, size, float_offset in self.ATTRIBUTES:
            glVertexAttribPointer(
                location,
                size,
                GL_FLOAT,
                GL_FALSE,
                self.STRIDE,
                ctypes.c_void_p(float_offset * 4),
            )
            glEnableVertexAttribArray(location)
        glBindVertexArray(0)
        self.count = 0

    def upload(self, vertices: np.ndarray, usage: GLenum = GL_DYNAMIC_DRAW) -> None:
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, usage)
        self.count = vertices.shape[0]

    def draw(self) -> None:
        if not self.count:
            return
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.count)

    def destroy(self) -> None:
        glDeleteBuffers(1, [self.vbo])
        glDeleteVertexArrays(1, [self.vao])


class Points:
    """A VAO/VBO for one GL_POINTS batch: ``x, y, z, r, g, b, radius_px``.

    One vertex per particle and the sprite is expanded by the GPU, so a field of
    particles costs one upload and one draw call — nothing is tessellated on the
    CPU. ``radius_px`` is the sprite radius in **pixels** (see
    ``POINT_VERTEX_SHADER``); ``OrbitCamera.pixel_radii`` is the world-size
    bridge. The default usage is ``GL_STREAM_DRAW`` because particle batches are
    rebuilt every frame.
    """

    STRIDE = 7 * 4  # xyz + rgb + radius_px, float32

    ATTRIBUTES = (
        (0, 3, 0),  # position
        (1, 3, 3),  # colour
        (2, 1, 6),  # sprite radius in pixels
    )

    def __init__(self) -> None:
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        if not self.vao or not self.vbo:
            raise RuntimeError("could not create the point buffers")
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        for location, size, float_offset in self.ATTRIBUTES:
            glVertexAttribPointer(
                location,
                size,
                GL_FLOAT,
                GL_FALSE,
                self.STRIDE,
                ctypes.c_void_p(float_offset * 4),
            )
            glEnableVertexAttribArray(location)
        glBindVertexArray(0)
        self.count = 0

    def upload(self, vertices: np.ndarray, usage: GLenum = GL_STREAM_DRAW) -> None:
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, usage)
        self.count = int(vertices.shape[0])

    def draw(self) -> None:
        if not self.count:
            return
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.count)

    def destroy(self) -> None:
        glDeleteBuffers(1, [self.vbo])
        glDeleteVertexArrays(1, [self.vao])
