"""Small OpenGL helpers: shader program building and a GL_LINES batch."""

from __future__ import annotations

import ctypes

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COMPILE_STATUS,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_LINES,
    GL_LINK_STATUS,
    GL_STATIC_DRAW,
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
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glLinkProgram,
    glShaderSource,
    glVertexAttribPointer,
)


def _compile_shader(source: str, kind: int) -> int:
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
    """A VAO/VBO for one GL_LINES batch. `upload` re-specifies the whole buffer."""

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

    def upload(self, vertices: np.ndarray, usage: int = GL_STATIC_DRAW) -> None:
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
        """Draw the uploaded vertices as GL_TRIANGLES instead of GL_LINES."""
        if not self.count:
            return
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.count)

    def destroy(self) -> None:
        glDeleteBuffers(1, [self.vbo])
        glDeleteVertexArrays(1, [self.vao])
