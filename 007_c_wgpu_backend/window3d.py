"""GLFW window lifecycle for the 3D experiment (wgpu edition).

``create_window`` opens a window **without** an OpenGL context
(``CLIENT_API = NO_API``): the GPU work goes through wgpu, which builds a Metal
(macOS), Vulkan (Linux) or DX12 (Windows) surface on the native window handle
in ``gpu3d.GPU``. GLFW is kept for what it is good at — the window, input and
the blocking event queue the reactive loop sleeps in.

There is no ``glViewport`` any more: the render pass always covers the whole
surface, and ``gpu3d`` resizes the swap-chain from the framebuffer size before
each frame. ``on_resize`` still exists so ``main.py`` can ask the reactive loop
for one repaint after a resize.
"""

from __future__ import annotations

import glfw
from settings3d import HEIGHT, TITLE, WIDTH


def create_window(
    title: str = TITLE,
    width: int = WIDTH,
    height: int = HEIGHT,
    on_resize=None,
):
    """Open a context-less window for wgpu and return its GLFW handle.

    ``on_resize(win, width, height)`` runs on framebuffer resizes (pixels), which
    is how ``main.py`` asks the reactive loop for one repaint: a resized window
    is a visible change, and nothing else would mark it dirty.
    """
    if not glfw.init():
        raise RuntimeError("glfw.init() failed: is DISPLAY / WAYLAND_DISPLAY set?")

    # wgpu owns the GPU side: no GL context, no swap interval, no GL version.
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("could not create a GLFW window")

    def handle_resize(win, new_width: int, new_height: int) -> None:
        if on_resize is not None:
            on_resize(win, new_width, new_height)

    glfw.set_framebuffer_size_callback(window, handle_resize)
    return window


def close_window(window) -> None:
    """Order matters: release the GPU surface first (gpu.close), then the window."""
    glfw.destroy_window(window)
    glfw.terminate()
