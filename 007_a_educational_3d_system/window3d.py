"""GLFW window lifecycle and screen helpers for the 3D experiment.

``create_window`` is step01's numbered setup collapsed into a function and
returns the raw GLFW handle, so ``main.py`` keeps the same plain
``while not glfw.window_should_close(window)`` loop as the 2D app.
"""

from __future__ import annotations

import glfw
from OpenGL.GL import glViewport
from settings3d import HEIGHT, TITLE, WIDTH


def create_window(
    title: str = TITLE,
    width: int = WIDTH,
    height: int = HEIGHT,
    vsync: bool = True,
):
    """Open an OpenGL 3.3 core window and return its GLFW handle."""
    if not glfw.init():
        raise RuntimeError("glfw.init() failed: is DISPLAY / WAYLAND_DISPLAY set?")

    for hint, value in (
        (glfw.CONTEXT_VERSION_MAJOR, 3),
        (glfw.CONTEXT_VERSION_MINOR, 3),
        (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE),
        (glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE),
    ):
        glfw.window_hint(hint, value)

    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("could not create an OpenGL 3.3 core context")

    glfw.make_context_current(window)
    glfw.swap_interval(1 if vsync else 0)

    def on_resize(_win, new_width: int, new_height: int) -> None:
        glViewport(0, 0, max(new_width, 1), max(new_height, 1))

    glfw.set_framebuffer_size_callback(window, on_resize)
    glViewport(0, 0, *glfw.get_framebuffer_size(window))
    return window


def close_window(window) -> None:
    """Order matters: destroy the window (and its context) before terminating GLFW."""
    glfw.destroy_window(window)
    glfw.terminate()
