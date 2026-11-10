"""Reactive redraw state: one flag that turns "redraw forever" into "redraw on change".

This is the 3D twin of ``experimental_probe/state.py``. Every GLFW callback that
changes what is on screen ends in ``mark()``. The loop (``loop3d.run``) blocks in
``glfw.wait_events_timeout`` until a callback fires, draws exactly one frame when
the flag is set, and goes back to sleep. With no input the process stays inside
the OS event queue: no frames, no GPU work, no CPU.

Only the generic "something happened" callbacks live here. Keys, scroll and the
macOS pinch bridge stay in ``main.py`` because they carry the app's logic
(reset, zoom, pan); each of them simply calls ``mark()`` when it has done
something visible. The framebuffer-size callback stays in ``window3d.py`` for
the same reason — it owns ``glViewport`` and accepts a resize hook.

Two details that matter:

* the loop draws only when this flag is set, so wake-ups that change nothing
  (focus, cursor enter/leave) cost a wake-up and nothing else;
* a repeated cursor position is not a change, so ``_on_cursor`` drops it — GLFW
  can resend the same coordinates, and redrawing for that would turn a reactive
  loop back into a polling one at event rate.
"""

from __future__ import annotations

import dataclasses

import glfw


@dataclasses.dataclass
class ReactiveState:
    """The redraw request plus the counters that prove the loop is reactive."""

    dirty: bool = True  # start dirty: the first frame must be drawn
    events: int = 0  # callbacks that asked for a redraw
    mouse: tuple[float, float] | None = None  # last cursor position, for de-duplication

    def mark(self) -> None:
        """Something visible changed: ask the loop for exactly one redraw."""
        self.dirty = True
        self.events += 1

    # ------------------------------------------------------------------ wiring
    def install(self, window) -> None:
        """Register the callbacks whose only job is to request a redraw."""
        glfw.set_cursor_pos_callback(window, self._on_cursor)
        glfw.set_mouse_button_callback(window, self._on_button)
        # Expose events: the window was uncovered or moved between monitors.
        glfw.set_window_refresh_callback(window, self._on_refresh)

    # --------------------------------------------------------------- callbacks
    def _on_cursor(self, _window, x: float, y: float) -> None:
        if (x, y) == self.mouse:
            return  # GLFW can resend the same position; nothing to show
        self.mouse = (x, y)
        self.mark()

    def _on_button(self, _window, _button: int, _action: int, _mods: int) -> None:
        # The camera polls button state while drawing, so the press/release
        # itself only has to make sure one frame is drawn to see it.
        self.mark()

    def _on_refresh(self, _window) -> None:
        self.mark()
