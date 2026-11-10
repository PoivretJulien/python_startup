"""Ready-made screen-space controls, built only from the public drawing API.

A control here is a small object that remembers its own state and has one
method, ``draw(app)``, to call once per frame from ``draw_frame``. That call
does everything — reads the mouse, updates the state, draws — and returns the
control's current value::

    import app3d
    from widgets2d import Slider

    app = app3d.App("my tool")
    volume = Slider((60, 200), (360, 200), 0, 100, value=50)

    def draw_frame():
        level = volume.draw(app)          # a float between 0 and 100
        ...

    app.run(draw_frame)

Nothing here touches the GPU directly: a slider is one ``draw_screen_line``, one
``draw_screen_circle`` and the hit tests of ``screen2d``. Read it as a worked
example of the drag recipe, then write your own controls the same way.
"""

from __future__ import annotations

import screen2d

GREY = (0.45, 0.45, 0.52, 1.0)
PURPLE = (0.58, 0.25, 0.90, 1.0)
YELLOW = (1.00, 0.85, 0.10, 1.0)


class Slider:
    """A line with a round handle; dragging the handle picks a value.

    The track goes from ``start`` to ``end`` — two ``(x, y)`` points in window
    pixels — so the same class makes a horizontal, a vertical or a slanted
    slider. The value is ``minimum`` at ``start`` and ``maximum`` at ``end``.

    Colours are ``(r, g, b)`` or ``(r, g, b, a)``:

    * ``line_color`` — the track,
    * ``handle_color`` — the handle at rest,
    * ``hover_color`` — the handle while the cursor is on it or it is dragged,
    * ``fill_color`` — optional: the part of the track from ``start`` to the
      handle, drawn over the track (``None`` = not drawn).

    ``thickness`` is the width of the track and ``radius`` the size of the
    handle, in pixels. ``step`` snaps the value: ``step=1`` gives whole
    numbers, ``step=0.25`` quarters, ``0`` (default) any float.

    What you can read or set at any time:

    * ``value`` — the current value (setting it moves the handle),
    * ``dragging`` — True while the user holds the handle: this is the flag
      that makes the handle follow the cursor,
    * ``hovered`` — True while the cursor is on the handle,
    * ``changed`` — True when the last ``draw`` call changed the value.
    """

    def __init__(
        self,
        start: screen2d.Point2,
        end: screen2d.Point2,
        minimum: float = 0.0,
        maximum: float = 1.0,
        value: float | None = None,
        *,
        line_color: screen2d.ColorRGBA = GREY,
        handle_color: screen2d.ColorRGBA = PURPLE,
        hover_color: screen2d.ColorRGBA = YELLOW,
        fill_color: screen2d.ColorRGBA | None = None,
        thickness: float = 4.0,
        radius: float = 10.0,
        step: float = 0.0,
    ) -> None:
        self.start = (float(start[0]), float(start[1]))
        self.end = (float(end[0]), float(end[1]))
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.line_color = line_color
        self.handle_color = handle_color
        self.hover_color = hover_color
        self.fill_color = fill_color
        self.thickness = float(thickness)
        self.radius = float(radius)
        self.step = float(step)

        self.dragging = False
        self.hovered = False
        self.changed = False
        self._grab = 0.0  # handle position minus cursor position, at the click
        self._was_down = False
        self._value = self._snap(self.minimum if value is None else float(value))

    # ------------------------------------------------------------------ value
    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, new: float) -> None:
        self._value = self._snap(float(new))

    def _snap(self, value: float) -> float:
        """Keep ``value`` between the two ends and, with a step, on a multiple."""
        low, high = sorted((self.minimum, self.maximum))
        if self.step > 0.0:
            value = self.minimum + round((value - self.minimum) / self.step) * self.step
        return min(max(value, low), high)

    # ------------------------------------------------------------- geometry
    # ``t`` is a position along the track: 0 at ``start``, 1 at ``end``.
    def _t_of_value(self) -> float:
        span = self.maximum - self.minimum
        return 0.0 if span == 0.0 else (self._value - self.minimum) / span

    def _t_of_point(self, point: screen2d.Point2) -> float:
        """Where ``point`` falls along the track (its shadow on the line)."""
        dx, dy = self.end[0] - self.start[0], self.end[1] - self.start[1]
        length_sq = dx * dx + dy * dy
        if length_sq <= 0.0:
            return 0.0
        return ((point[0] - self.start[0]) * dx + (point[1] - self.start[1]) * dy) / length_sq

    def handle(self) -> tuple[float, float]:
        """The centre of the handle, in window pixels."""
        t = self._t_of_value()
        return (
            self.start[0] + (self.end[0] - self.start[0]) * t,
            self.start[1] + (self.end[1] - self.start[1]) * t,
        )

    # ------------------------------------------------------------------ frame
    def draw(self, app) -> float:
        """Read the mouse, move the handle, draw the slider; return the value.

        Call it once per frame, from ``draw_frame``. ``app`` is the
        ``app3d.App``: it gives the cursor, the button and the renderer.
        """
        ui = app.aa
        cursor = app.cursor()
        down = app.mouse_down()
        pressed = down and not self._was_down  # true in one frame: the click
        self._was_down = down

        handle = self.handle()
        on_handle = screen2d.in_circle(cursor, handle, self.radius)

        # --- the drag: beginning, end, middle ----------------------------
        if pressed:
            if on_handle:
                # Grabbed by the handle: remember where, so it does not jump.
                self.dragging = True
                self._grab = self._t_of_value() - self._t_of_point(cursor)
            elif screen2d.in_line(cursor, self.start, self.end, 2.0 * self.radius):
                # Clicked on the track: the handle comes to the cursor.
                self.dragging = True
                self._grab = 0.0
        if not down:
            self.dragging = False

        before = self._value
        if self.dragging:
            t = min(max(self._t_of_point(cursor) + self._grab, 0.0), 1.0)
            self._value = self._snap(self.minimum + (self.maximum - self.minimum) * t)
            handle = self.handle()
        self.changed = self._value != before

        # A button held since somewhere else (another slider being dragged)
        # must not light this handle up when the cursor passes over it.
        self.hovered = self.dragging or (on_handle and not down)

        # --- the picture: track, optional fill, handle on top -----------------
        ui.draw_screen_line(self.start, self.end, self.line_color, self.thickness)
        if self.fill_color is not None:
            ui.draw_screen_line(self.start, handle, self.fill_color, self.thickness)
        ui.draw_screen_circle(
            handle, self.radius, self.hover_color if self.hovered else self.handle_color
        )
        return self._value
