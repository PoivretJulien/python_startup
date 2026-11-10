"""The window + GPU + reactive loop of ``main.py``, packed for small programs.

``main.py`` wires everything by hand, which is the right thing to read once.
A small tool only wants the last part — a ``draw_frame()`` — so this module
does the wiring and nothing else::

    import app3d

    app = app3d.App("my tool", size=(640, 420))

    def draw_frame() -> None:
        over = app.aa.in_screen_circle(app.cursor(), (320, 210), 60)
        app.aa.draw_screen_circle((320, 210), 60, (1, 1, 0) if over else (0.6, 0.2, 0.9))

    app.run(draw_frame)

Nothing in the existing modules changes: ``App`` only calls them, in the order
``main.py`` does. What you get:

* ``app.aa`` / ``app.plain`` — the two renderers (same drawing API),
* ``app.camera`` — the shared ``OrbitCamera`` (call ``app.camera.handle(app.window)``
  inside ``draw_frame`` if the tool wants orbit / pan / zoom),
* ``app.cursor()`` / ``app.mouse_down()`` — the input a screen-space tool needs,
* ``app.request_redraw()`` — ask for one frame from anywhere, **including
  another thread** (a network message arrived, a file changed ...),
* ``app.args`` — the parsed command line.

The loop is the reactive one by default: no event, no frame, no CPU. The
command line of ``main.py`` is kept: ``--poll`` (always redraw), ``--refresh
HZ``, ``--wait SECONDS``, ``--backend``, ``--no-vsync``, ``--seconds``. Pass
your own ``argparse`` parser to add options next to them.
"""

from __future__ import annotations

import argparse
import signal
import time
from collections.abc import Callable, Sequence

import glfw
import loop3d
import reactive3d
from camera3d import OrbitCamera
from gpu3d import GPU
from renderer3d import GridArrowRenderer
from renderer_aa3d import AAGridArrowRenderer
from settings3d import BACKGROUND, DEFAULT_WAIT_TIMEOUT, HEIGHT, IDLE_REFRESH_HZ, TITLE, WIDTH


def add_loop_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The loop's knobs, the same ones as ``main.py`` — reactive unless ``--poll``."""
    group = parser.add_argument_group("loop")
    group.add_argument(
        "--poll",
        action="store_true",
        help="naive busy loop: poll + redraw every iteration (default: reactive)",
    )
    group.add_argument(
        "--refresh",
        type=float,
        default=IDLE_REFRESH_HZ,
        metavar="HZ",
        help="additionally redraw at this rate (0 = pure reactive)",
    )
    group.add_argument(
        "--wait",
        type=float,
        default=DEFAULT_WAIT_TIMEOUT,
        metavar="SECONDS",
        help="max block time in wait_events (0 = block forever)",
    )
    group.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "metal", "vulkan", "dx12", "gl"),
        help="wgpu backend (auto = Metal on macOS, Vulkan on Linux, DX12 on Windows)",
    )
    group.add_argument("--no-vsync", action="store_true", help="present without waiting for vsync")
    group.add_argument(
        "--seconds", type=float, default=0.0, help="auto-exit after N seconds (0 = never)"
    )
    return parser


class App:
    """One window, its GPU, both renderers and the reactive loop."""

    def __init__(
        self,
        title: str = TITLE,
        size: tuple[int, int] = (WIDTH, HEIGHT),
        position: tuple[int, int] | None = None,
        *,
        background: Sequence[float] = BACKGROUND,
        parser: argparse.ArgumentParser | None = None,
        argv: Sequence[str] | None = None,
    ) -> None:
        if parser is None:
            parser = argparse.ArgumentParser(description=title)
        if not any(action.dest == "poll" for action in parser._actions):
            add_loop_args(parser)
        self.args = parser.parse_args(argv)

        # Every callback that changes the picture calls reactive.mark(); the
        # loop then draws exactly one frame.
        self.reactive = reactive3d.ReactiveState()
        self.window = glfw_window = self._open(title, size, position)
        self.reactive.install(glfw_window)

        self.gpu = GPU(glfw_window, backend=self.args.backend, vsync=not self.args.no_vsync)
        self.gpu.clear_color = (*background[:3], 1.0)
        self.camera = OrbitCamera()
        self.plain = GridArrowRenderer(self.gpu, self.camera)
        self.aa = AAGridArrowRenderer(self.gpu, self.camera)

        self._running = True
        glfw.set_key_callback(glfw_window, self._on_key)

    def _open(self, title, size, position):
        from window3d import create_window

        window = create_window(
            title, int(size[0]), int(size[1]), on_resize=lambda _w, _x, _y: self.reactive.mark()
        )
        if position is not None:
            glfw.set_window_pos(window, int(position[0]), int(position[1]))
        return window

    def _on_key(self, window, key: int, _scancode: int, action: int, _mods: int) -> None:
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    # ------------------------------------------------------------------ input
    def cursor(self) -> tuple[float, float]:
        """Cursor position in window pixels, top-left origin — screen-space units."""
        return glfw.get_cursor_pos(self.window)

    def mouse_down(self, button: int = glfw.MOUSE_BUTTON_LEFT) -> bool:
        """True while the mouse button is held."""
        return glfw.get_mouse_button(self.window, button) == glfw.PRESS

    def size(self) -> tuple[int, int]:
        """Window size in the same pixels as ``cursor()``."""
        return glfw.get_window_size(self.window)

    # ------------------------------------------------------------------- loop
    def request_redraw(self) -> None:
        """Ask for one frame. Safe from any thread.

        ``mark()`` raises the dirty flag; ``post_empty_event`` wakes the loop,
        which is asleep inside the OS event queue in reactive mode. Without the
        wake-up a change made by another thread would only be seen at the next
        mouse move (or ``--wait`` timeout).
        """
        self.reactive.mark()
        glfw.post_empty_event()

    def quit(self) -> None:
        """Leave ``run()`` at the next loop turn. Safe from any thread."""
        self._running = False
        glfw.post_empty_event()

    def run(self, draw_frame: Callable[[], None]) -> loop3d.Stats:
        """Run ``draw_frame`` once per visible change until the window closes.

        ``draw_frame`` only issues draw calls: the loop clears, presents and
        sleeps. Esc, the close button, Ctrl+C and ``--seconds`` end the run;
        everything is released before this returns.
        """
        args = self.args

        def on_sigint(_signum, _frame) -> None:
            self._running = False

        previous = signal.signal(signal.SIGINT, on_sigint)
        deadline = time.perf_counter() + args.seconds if args.seconds > 0.0 else None

        def should_stop() -> bool:
            if not self._running:
                return True
            return deadline is not None and time.perf_counter() >= deadline

        # A deadline can only expire if the loop wakes on its own.
        wait = args.wait if args.wait > 0.0 or deadline is None else 0.05
        try:
            stats = loop3d.run(
                self.gpu,
                self.reactive,
                draw_frame,
                poll=args.poll,
                refresh=args.refresh,
                wait=wait,
                should_stop=should_stop,
            )
        finally:
            signal.signal(signal.SIGINT, previous)
            self.close()
        print(stats.summary())
        return stats

    def close(self) -> None:
        from window3d import close_window

        if self.window is None:
            return
        self.aa.close()
        self.plain.close()
        self.gpu.close()
        close_window(self.window)
        self.window = None
