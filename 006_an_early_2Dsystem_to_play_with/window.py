"""GLFW window lifecycle and input helpers, kept close to step01.

`create_window()` is step01's numbered setup (init -> hints -> create -> current
-> viewport) collapsed into a function. It returns the raw GLFW handle, so main
keeps step01's plain `while not glfw.window_should_close(window)` loop.

`close_window()` reverses it: destroy the window before terminating GLFW.
"""

from __future__ import annotations

import geometry
import glfw
import settings
from OpenGL.GL import glViewport


def create_window(
    title: str = settings.TITLE,
    width: int = settings.WIDTH,
    height: int = settings.HEIGHT,
):
    """Open an OpenGL 3.3 core window and return its GLFW handle."""
    if not glfw.init():
        raise RuntimeError("glfw.init() failed: is DISPLAY / WAYLAND_DISPLAY set?")

    # Ask for a specific OpenGL version *before* the window exists.
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
    glfw.swap_interval(1)

    # Resizes come as callbacks; the framebuffer size is what the viewport needs.
    def on_resize(win, width: int, height: int) -> None:
        glViewport(0, 0, max(width, 1), max(height, 1))

    glfw.set_framebuffer_size_callback(window, on_resize)
    glViewport(0, 0, *glfw.get_framebuffer_size(window))
    return window


def close_window(window) -> None:
    """Order matters: destroy the window (and its context) before terminating GLFW."""
    glfw.destroy_window(window)
    glfw.terminate()


def cursor_world(window) -> geometry.Point:
    """Cursor in window pixels -> the same 2D world the renderer uses."""
    px, py = glfw.get_cursor_pos(window)
    width, height = glfw.get_window_size(window)
    return geometry.screen_to_world(px, py, width, height)
