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

``Slider`` picks a value; ``Ruler`` is a measuring overlay for laying out an
interface (also available in any program with the ``--ruler`` option).

Nothing here touches the GPU directly: a slider is one ``draw_screen_line``, one
``draw_screen_circle`` and the hit tests of ``screen2d``. Read it as a worked
example of the drag recipe, then write your own controls the same way.
"""

from __future__ import annotations

import math

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


class Ruler:
    """A measuring overlay: where the cursor is, and how far it was dragged.

    A tool for *you*, while laying out an interface. Draw it last, so it is on
    top of everything::

        ruler = Ruler()

        def draw_frame():
            ...your interface...
            ruler.draw(app)

    or do not write anything: every program built on ``app3d.App`` shows it when
    started with ``--ruler`` (and F1 switches it on and off while running).

    What it shows, all in window pixels — the units of every screen method:

    * always: a crosshair through the cursor and its position ``x, y``;
    * while the mouse button is held: the point where the click happened
      (``from``), a line from there to the cursor with its length ``d``, and the rectangle
      between the two points with its width ``dx`` and height ``dy`` — which
      are exactly the ``anchor`` and ``size`` of ``draw_screen_rect``.

    The ruler only watches the mouse; the program underneath still receives the
    same clicks. After ``draw`` these can be read:

    * ``measuring`` — True while the button is held,
    * ``start`` — where the click happened, ``end`` — the cursor,
    * ``distance`` — the length between them, in pixels (0.0 when not measuring).
    """

    def __init__(
        self,
        color: screen2d.ColorRGBA = (1.00, 0.55, 0.10, 1.0),
        text_color: screen2d.ColorRGBA = (1.00, 1.00, 1.00),
        panel_color: screen2d.ColorRGBA = (0.05, 0.05, 0.07, 0.85),
        text_size: float = 12.0,
    ) -> None:
        self.color = screen2d.rgba(color)
        self.text_color = text_color
        self.panel_color = panel_color
        self.text_size = float(text_size)
        self.measuring = False
        self.start = (0.0, 0.0)
        self.end = (0.0, 0.0)
        self.distance = 0.0
        self._was_down = False

    def _label(self, app, near: screen2d.Point2, lines: list[str]) -> None:
        """A small text panel next to ``near``, kept inside the window."""
        ui = app.aa
        size = self.text_size
        pad, gap = 7.0, 5.0
        width = max(ui.measure_screen_text(line, size) for line in lines) + 2 * pad
        height = len(lines) * size + (len(lines) - 1) * gap + 2 * pad
        window_w, window_h = app.size()
        x, y = near[0] + 16, near[1] + 18  # below-right of the point ...
        if x + width > window_w:
            x = near[0] - 16 - width  # ... or on the other side near an edge
        if y + height > window_h:
            y = near[1] - 18 - height
        x, y = max(x, 0.0), max(y, 0.0)
        ui.draw_screen_rect((x, y), (width, height), self.panel_color, 6)
        for index, line in enumerate(lines):
            baseline = y + pad + size + index * (size + gap)
            ui.draw_screen_text(x + pad, baseline, line, size, self.text_color)

    def draw(self, app) -> float:
        """Draw the overlay for this frame; return the measured distance.

        Call it once per frame, after everything else. ``app`` is the
        ``app3d.App``.
        """
        ui = app.aa
        cursor = app.cursor()
        down = app.mouse_down()
        if down and not self._was_down:
            self.start = (float(cursor[0]), float(cursor[1]))  # the click
        self._was_down = down
        self.measuring = down
        self.end = (float(cursor[0]), float(cursor[1]))

        r, g, b, _a = self.color
        faint = (r, g, b, 0.45)
        window_w, window_h = app.size()

        # The crosshair: one dashed line across the window in each direction.
        ui.draw_screen_dashed_line((0, cursor[1]), (window_w, cursor[1]), faint, 1, 6, 5)
        ui.draw_screen_dashed_line((cursor[0], 0), (cursor[0], window_h), faint, 1, 6, 5)

        if not self.measuring:
            self.distance = 0.0
            self._label(app, cursor, [f"x {cursor[0]:.0f}   y {cursor[1]:.0f}"])
            return 0.0

        sx, sy = self.start
        dx, dy = cursor[0] - sx, cursor[1] - sy
        self.distance = math.hypot(dx, dy)

        # The rectangle between the two points: its top-left corner and size
        # are what draw_screen_rect calls `anchor` and `size`.
        anchor = (min(sx, cursor[0]), min(sy, cursor[1]))
        size = (abs(dx), abs(dy))
        if size[0] >= 1.0 and size[1] >= 1.0:
            ui.draw_screen_rect(anchor, size, (r, g, b, 0.10))
            ui.draw_screen_rect_outline(anchor, size, faint, 0, 1)
        ui.draw_screen_line(self.start, cursor, self.color, 2)  # the measured length
        ui.draw_screen_circle_outline(self.start, 6, self.color, 2)  # where the click was
        ui.draw_screen_circle(cursor, 3, self.color)

        self._label(
            app,
            cursor,
            [
                f"x {cursor[0]:.0f}   y {cursor[1]:.0f}",
                f"from ({sx:.0f}, {sy:.0f})   d {self.distance:.1f}",
                f"dx {dx:+.0f}   dy {dy:+.0f}",
                f"anchor ({anchor[0]:.0f}, {anchor[1]:.0f})   size ({size[0]:.0f}, {size[1]:.0f})",
            ],
        )
        return self.distance
