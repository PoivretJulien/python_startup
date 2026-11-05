"""The reactive event loop: redraw when something changed, otherwise sleep.

    wait_events_timeout(wait)  ->  callbacks mark dirty  ->  draw once  ->  swap

This is the 3D twin of ``experimental_probe/loop.py``, and it is the whole API
change: ``main`` hands over a ``draw`` callable that only issues GL calls, and
the loop owns the event pump, the dirty check, the buffer swap and the frame
count. The scene, the camera and the renderers never learn that the loop
changed.

With no input the process blocks inside the OS event queue: no frames, no CPU.
The naive alternative (``poll=True``) polls and redraws unconditionally, which
runs flat out forever — both modes call the same ``draw``, so the difference is
measurable instead of a matter of faith.

Three details that matter in a real tool:

* wake-ups that change nothing do not redraw: ``state.dirty`` is the only
  trigger, so "reactive" never degenerates into polling at event rate;
* ``wait`` is a cap on the block time, not the frame rate: it lets ``--seconds``
  and Ctrl+C expire. Measured on X11, a Python SIGINT handler does not run
  while the process sits inside ``wait_events()`` (XNextEvent swallows EINTR),
  so ``wait=0`` gives the best battery life but delays terminal Ctrl+C until
  the next window event. Esc and the close button are real events and always
  work;
* ``refresh`` adds a deliberate tick for time-based scenes (an animated clock,
  for instance), which pure change-detection can never see.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable

import glfw
from reactive3d import ReactiveState
from settings3d import DEFAULT_WAIT_TIMEOUT


@dataclasses.dataclass
class Stats:
    """What one run of the loop did, printed on exit so the claim is checkable.

    ``cpu_seconds`` is ``time.process_time()``, which counts CPU time and ignores
    time spent asleep: frames drop to almost zero while CPU stays near zero too.
    """

    mode: str = "REACTIVE"
    frames: int = 0
    events: int = 0
    elapsed: float = 0.0
    cpu_seconds: float = 0.0

    @property
    def cpu_percent(self) -> float:
        return 100.0 * self.cpu_seconds / self.elapsed if self.elapsed > 0.0 else 0.0

    @property
    def redraws_per_second(self) -> float:
        return self.frames / self.elapsed if self.elapsed > 0.0 else 0.0

    def summary(self) -> str:
        # Below ~0.25 s the CPU ratio is interpreter startup noise, so report the
        # absolute CPU time without a misleading percentage.
        cpu = f"CPU {self.cpu_seconds:.3f} s"
        if self.elapsed >= 0.25:
            cpu += f" ({self.cpu_percent:.1f}%)"
        return (
            f"  {self.mode:<8} · {self.frames} redraws in {self.elapsed:.2f} s "
            f"({self.redraws_per_second:.1f}/s) · {self.events} events · {cpu}"
        )


def run(
    window,
    state: ReactiveState,
    draw: Callable[[], None],
    *,
    poll: bool = False,
    refresh: float = 0.0,
    wait: float = DEFAULT_WAIT_TIMEOUT,
    should_stop: Callable[[], bool] | None = None,
) -> Stats:
    """Present one frame per visible change and block in GLFW in between.

    ``draw`` clears the screen and issues the draw calls; it must not poll
    events or swap buffers — the loop does both, once per redraw, right after
    the dirty flag was cleared (so callbacks fired while drawing are not lost).
    ``refresh`` is a redraw rate in Hz (0 = pure reactive), and ``should_stop``
    is polled between blocks for Ctrl+C and ``--seconds`` — which is why
    ``wait`` must stay finite when those matter (0 blocks forever).
    """
    stats = Stats(mode="POLL" if poll else "REACTIVE")
    started = time.perf_counter()
    cpu_started = time.process_time()
    tick = 1.0 / refresh if refresh > 0.0 else 0.0
    last_tick = 0.0

    def finished() -> bool:
        if glfw.window_should_close(window):
            return True
        return should_stop is not None and should_stop()

    while not finished():
        if poll:
            glfw.poll_events()
            redraw = True  # the naive loop: always redraw
        else:
            _wait(wait, tick)
            redraw = state.dirty  # ... and only draw if it mattered

        if finished():
            break

        now = time.perf_counter()
        stats.elapsed = now - started  # live values for the readout
        if tick and now - last_tick >= tick:
            redraw = True  # --refresh: animate on purpose
            last_tick = now
        if not redraw:
            continue  # spurious wake-up, nothing to show

        state.dirty = False  # cleared *before* drawing
        draw()
        glfw.swap_buffers(window)
        stats.frames += 1

    stats.elapsed = time.perf_counter() - started
    stats.cpu_seconds = time.process_time() - cpu_started
    stats.events = state.events
    return stats


def _wait(wait: float, tick: float) -> None:
    """Block for events, but no longer than the next tick or the wait cap."""
    cap = _block_seconds(wait, tick)
    if cap is None:
        glfw.wait_events()  # block until an event arrives
    else:
        glfw.wait_events_timeout(cap)


def _block_seconds(wait: float, tick: float) -> float | None:
    """How long ``wait_events`` may block: None = forever, otherwise seconds > 0."""
    caps = [value for value in (wait, tick) if value > 0.0]
    return min(caps) if caps else None
