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
* ``app.keys_down()`` / ``app.key_down(name)`` / ``app.key_pressed(name)`` /
  ``app.typed()`` — the keyboard: what is held, what was just pressed, and the
  text that was typed,
* ``app.pan()`` / ``app.zoom()`` — the pan and zoom gestures (trackpad pinch
  and two-finger scroll, mouse wheel), the same on macOS, Windows and Linux,
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

import gestures3d
import glfw
import loop3d
import reactive3d
from camera3d import OrbitCamera
from gpu3d import GPU
from renderer3d import GridArrowRenderer
from renderer_aa3d import AAGridArrowRenderer
from settings3d import BACKGROUND, DEFAULT_WAIT_TIMEOUT, HEIGHT, IDLE_REFRESH_HZ, TITLE, WIDTH


def _special_key_names() -> dict[int, str]:
    """GLFW key code -> a readable name, for the keys that print nothing.

    Built from GLFW's own constants, so it is complete without a hand-written
    table: ``KEY_LEFT_SHIFT`` becomes ``"LEFT SHIFT"``, ``KEY_F5`` ``"F5"``,
    ``KEY_KP_ADD`` ``"KP ADD"`` (KP = the numeric keypad).
    """
    names = {}
    for attribute in dir(glfw):
        if attribute.startswith("KEY_") and attribute not in ("KEY_UNKNOWN", "KEY_LAST"):
            code = getattr(glfw, attribute)
            if isinstance(code, int):
                names[code] = attribute[4:].replace("_", " ")
    return names


KEY_NAMES = _special_key_names()


def key_name(key: int, scancode: int = 0) -> str:
    """The name of a key, as the methods of ``App`` report it.

    A key that prints a character is named by that character, in upper case and
    **as printed on the user's keyboard**: on a French AZERTY layout the key
    left of Z is ``"A"``, not the ``"Q"`` of the American layout GLFW's
    constants are named after. Every other key gets a word: ``"SPACE"``,
    ``"ENTER"``, ``"BACKSPACE"``, ``"TAB"``, ``"ESCAPE"``, ``"LEFT"``,
    ``"RIGHT"``, ``"UP"``, ``"DOWN"``, ``"LEFT SHIFT"``, ``"RIGHT CONTROL"``,
    ``"LEFT ALT"``, ``"LEFT SUPER"`` (the Command / Windows key), ``"F1"`` ...
    """
    if glfw.KEY_APOSTROPHE <= key <= glfw.KEY_WORLD_2:  # the printable block
        printed = glfw.get_key_name(key, scancode)
        if printed:
            return printed.upper()
    return KEY_NAMES.get(key, f"KEY {key}")


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
        # Keyboard state. GLFW reports each change once, through a callback;
        # these keep what the callbacks said so draw_frame can simply ask.
        self._held: dict[int, str] = {}  # key code -> name, in press order
        self._pending_pressed: list[tuple[str, bool]] = []  # (name, is a repeat)
        self._pending_typed: list[str] = []
        self._pressed: tuple[tuple[str, bool], ...] = ()  # what the current frame sees
        self._typed = ""
        glfw.set_key_callback(glfw_window, self._on_key)
        glfw.set_char_callback(glfw_window, self._on_char)
        # Pan / zoom gestures, collected like the key presses: the callbacks
        # add up what happened, the next frame receives the total.
        self._pending_pan = [0.0, 0.0]
        self._pending_zoom = 1.0
        self._pan = (0.0, 0.0)
        self._zoom = 1.0
        self.gestures = gestures3d.install(glfw_window, self._on_pan, self._on_zoom)

    def _open(self, title, size, position):
        from window3d import create_window

        window = create_window(
            title, int(size[0]), int(size[1]), on_resize=lambda _w, _x, _y: self.reactive.mark()
        )
        if position is not None:
            glfw.set_window_pos(window, int(position[0]), int(position[1]))
        return window

    def _on_key(self, window, key: int, scancode: int, action: int, _mods: int) -> None:
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            name = key_name(key, scancode)
            self._held[key] = name
            self._pending_pressed.append((name, False))
        elif action == glfw.RELEASE:
            self._held.pop(key, None)
        else:
            # REPEAT: the key is still held and the system "presses" it again,
            # as in any text field. Reported only to who asks for repeats.
            self._pending_pressed.append((self._held.get(key) or key_name(key, scancode), True))
        self.reactive.mark()  # the keyboard changed: one frame, please

    def _on_char(self, _window, codepoint: int) -> None:
        # A *character* was produced: layout, Shift, dead keys and auto-repeat
        # are already resolved by the system. This is what a text field wants.
        self._pending_typed.append(chr(codepoint))
        self.reactive.mark()

    def _on_pan(self, dx: float, dy: float) -> None:
        self._pending_pan[0] += dx
        self._pending_pan[1] += dy
        self.reactive.mark()

    def _on_zoom(self, factor: float) -> None:
        self._pending_zoom *= factor  # zooms multiply: two steps of 10 % make 21 %
        self.reactive.mark()

    def _begin_frame_input(self) -> None:
        """Hand the events collected since the last frame to the frame starting now."""
        self._pan = (self._pending_pan[0], self._pending_pan[1])
        self._zoom = self._pending_zoom
        self._pending_pan[:] = (0.0, 0.0)
        self._pending_zoom = 1.0
        self._pressed = tuple(self._pending_pressed)
        self._typed = "".join(self._pending_typed)
        self._pending_pressed.clear()
        self._pending_typed.clear()

    # ------------------------------------------------------------------ input
    def cursor(self) -> tuple[float, float]:
        """Cursor position in window pixels, top-left origin — screen-space units."""
        return glfw.get_cursor_pos(self.window)

    def mouse_down(self, button: int = glfw.MOUSE_BUTTON_LEFT) -> bool:
        """True while the mouse button is held."""
        return glfw.get_mouse_button(self.window, button) == glfw.PRESS

    def keys_down(self) -> list[str]:
        """Names of the keys held right now, in the order they were pressed.

        See ``key_name`` for the names: ``"A"``, ``"7"``, ``"SPACE"``,
        ``"LEFT SHIFT"``, ``"UP"`` ... An empty list when nothing is held.
        """
        return list(self._held.values())

    def key_down(self, name: str) -> bool:
        """True while the key called ``name`` is held (``app.key_down("SPACE")``).

        Like ``mouse_down``: true on every frame of the hold. ``"SHIFT"``,
        ``"CONTROL"``, ``"ALT"`` and ``"SUPER"`` match either side.
        """
        name = name.upper()
        held = self._held.values()
        return name in held or f"LEFT {name}" in held or f"RIGHT {name}" in held

    def keys_pressed(self, repeat: bool = False) -> tuple[str, ...]:
        """Names of the keys pressed since the previous frame, oldest first.

        Each press is reported in exactly one frame, however long the key is
        then held — the keyboard twin of a mouse click. ``repeat=True`` also
        reports the system's auto-repeat of a held key, which is what
        Backspace or an arrow key should do in a text field.
        """
        return tuple(name for name, repeated in self._pressed if repeat or not repeated)

    def key_pressed(self, name: str, repeat: bool = False) -> bool:
        """True in the one frame that follows a press of ``name``.

        With ``repeat=True`` it is true again each time a held key auto-repeats.
        """
        return name.upper() in self.keys_pressed(repeat)

    def typed(self) -> str:
        """The text typed since the previous frame (often ``""`` or one character).

        Characters, not keys: Shift, the keyboard layout, accents and
        auto-repeat are already applied (holding ``A`` keeps producing
        ``"a"``). Keys that print nothing (Enter, Backspace, arrows) are not
        here: ask ``key_pressed`` for those.
        """
        return self._typed

    def pan(self) -> tuple[float, float]:
        """The pan gesture since the previous frame, as ``(dx, dy)`` pixels.

        Add it to the position of what you are showing and the content follows
        the fingers. ``(0.0, 0.0)`` when nothing happened. It comes from a
        two-finger scroll on a trackpad, or Shift + mouse wheel; see
        ``gestures3d`` for what each system provides, and ``app.gestures`` for
        a one-line description of this one.
        """
        return self._pan

    def zoom(self) -> float:
        """The zoom gesture since the previous frame, as a factor to multiply by.

        ``1.0`` when nothing happened; above 1 to come closer, below 1 to move
        away. It comes from a trackpad pinch, the mouse wheel, or Ctrl + scroll.
        """
        return self._zoom

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
            def frame() -> None:
                self._begin_frame_input()  # this frame's key presses and text
                draw_frame()

            stats = loop3d.run(
                self.gpu,
                self.reactive,
                frame,
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
