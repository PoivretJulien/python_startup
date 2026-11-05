"""GLFW window lifecycle and screen helpers for the 3D experiment.

``create_window`` is step01's numbered setup collapsed into a function and
returns the raw GLFW handle. ``main.py`` no longer keeps the plain
``while not glfw.window_should_close(window)`` loop: the event pump and the
buffer swap live in ``loop3d.run``, and this module only has to keep the
viewport correct and forward resizes to the reactive state.
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
    on_resize=None,
):
    """Open an OpenGL 3.3 core window and return its GLFW handle.

    ``on_resize(win, width, height)`` runs after the viewport update, which is
    how ``main.py`` asks the reactive loop for one repaint: a resized window is
    a visible change, and nothing else would mark it dirty.
    """
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

    def handle_resize(win, new_width: int, new_height: int) -> None:
        glViewport(0, 0, max(new_width, 1), max(new_height, 1))
        if on_resize is not None:
            on_resize(win, new_width, new_height)

    glfw.set_framebuffer_size_callback(window, handle_resize)
    glViewport(0, 0, *glfw.get_framebuffer_size(window))
    return window


def close_window(window) -> None:
    """Order matters: destroy the window (and its context) before terminating GLFW."""
    glfw.destroy_window(window)
    glfw.terminate()
