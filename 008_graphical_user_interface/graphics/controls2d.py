"""Controls to click and read: a button and a table (mixed into both renderers).

Both are one method, called from ``draw_frame`` like any shape::

    if ui.draw_button((20, 20), (120, 34), "Save"):
        save()                                    # runs once per click

    ui.draw_table((20, 80), (400, 260), rows, WHITE, [DARK, DARKER])

A button needs no object and no callback: the call draws it, lights it up
while the cursor is over it, and answers ``True`` in the frame that follows a
click. A page of an interface is then plain Python — ``if`` a button was
clicked, change a variable; draw what the variable says.

Energy: nothing here runs on a timer. The window sleeps until an event
arrives (a mouse move, a click, a key); the frame it then draws costs one
draw call for the body of a button, and three for a whole table whatever its
number of lines — the table's geometry is built once and remembered until its
content changes.

The mouse is known through ``ui.pointer``, which ``app3d.App`` fills before
each frame from the window's events. A click is a press and a release that
both happened on the control, so pressing, sliding away and releasing cancels.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import screen2d
from charts2d import color_list

ACCENT = (0.05, 0.52, 0.57, 1.0)  # the green-blue of the controls
BUTTON = (0.02, 0.17, 0.19, 1.0)  # a button at rest: the accent, dark
BUTTON_HOVER = ACCENT  # under the cursor, and the colour to give a button that is "on"
BUTTON_TEXT = ACCENT  # the name, on a button at rest
BUTTON_HOVER_TEXT = (0.0, 0.0, 0.0, 1.0)  # the name, on the accent: black
TABLE_TEXT = ACCENT
TABLE_ROWS = ((0.03, 0.07, 0.08, 1.0), (0.04, 0.11, 0.12, 1.0))  # two backgrounds, in turn
TABLE_HEADER = (0.02, 0.17, 0.19, 1.0)
TABLE_HEADER_TEXT = (0.10, 0.72, 0.78)
FAR = (-1.0e9, -1.0e9)  # a cursor that is on nothing


class Pointer:
    """What the mouse did since the previous frame.

    * ``cursor`` — where it is, in window pixels;
    * ``press`` — where the left button went down, while it is held (else ``None``);
    * ``clicks`` — the clicks finished since the previous frame, each one
      ``(where pressed, where released)``. Usually empty, sometimes one.
    """

    def __init__(self) -> None:
        self.cursor: tuple[float, float] = FAR
        self.press: tuple[float, float] | None = None
        self.clicks: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = ()
        self.redraw: Callable[[], None] = lambda: None  # asks for one more frame

    def over(self, anchor, size, radius: float = 0.0) -> bool:
        """True when the cursor is on the rectangle — and no drag begun elsewhere passes by."""
        if not screen2d.in_rect(self.cursor, anchor, size, radius):
            return False
        return self.press is None or screen2d.in_rect(self.press, anchor, size, radius)

    def held(self, anchor, size, radius: float = 0.0) -> bool:
        """True while the button is held down on the rectangle."""
        return self.press is not None and self.over(anchor, size, radius)

    def clicked(self, anchor=None, size=None, radius: float = 0.0) -> bool:
        """True in the frame after a click — anywhere, or inside the rectangle given."""
        if anchor is None:
            return bool(self.clicks)
        return any(
            screen2d.in_rect(pressed, anchor, size, radius) and screen2d.in_rect(released, anchor, size, radius)
            for pressed, released in self.clicks
        )


def shade(color, factor: float) -> tuple[float, float, float, float]:
    """The same colour, darker (factor < 1) or lighter (> 1)."""
    r, g, b, a = screen2d.rgba(color)
    return (min(r * factor, 1.0), min(g * factor, 1.0), min(b * factor, 1.0), a)


def fit_text(measure, text: str, size: float, room: float) -> str:
    """``text``, cut and ended with ``..`` when it is wider than ``room`` pixels."""
    if measure(text, size) <= room:
        return text
    while text and measure(text + "..", size) > room:
        text = text[:-1]
    return text + ".." if text else ""


class ControlMethods:
    """``draw_button`` and ``draw_table``. Needs the screen pipelines of the renderer."""

    pointer = Pointer()  # replaced by the App's own; this one never sees a mouse

    # --------------------------------------------------------------- button
    def draw_button(
        self,
        anchor: screen2d.Point2,
        size: screen2d.Size2,
        name: str,
        color: screen2d.ColorRGBA = BUTTON,
        hover_color: screen2d.ColorRGBA = BUTTON_HOVER,
        text_color: screen2d.ColorRGBA = BUTTON_TEXT,
        *,
        hover_text_color: screen2d.ColorRGBA = BUTTON_HOVER_TEXT,
        border_color: screen2d.ColorRGBA | None = None,
        blur: float = 0.0,
        radius: float = 4.0,
        text_size: float = 12.0,
        enabled: bool = True,
    ) -> bool:
        """Draw a button with its ``name`` on it; return ``True`` when it was clicked.

        ``anchor`` is the top-left corner and ``size`` is ``(width, height)``,
        in window pixels. ``color`` is the button at rest, ``hover_color`` the
        button while the cursor is over it (a little darker while pressed), and
        ``text_color`` the name, which is centred and shortened with ``..`` if
        it does not fit. ``hover_text_color`` is the name while the cursor is
        over the button (black by default, to read on the accent). ``radius``
        rounds the corners. ``border_color`` adds a line one pixel wide around
        the button — what makes a white button stand on a light background.
        ``blur`` (pixels) makes the button of frosted glass: what was drawn
        before it shows through, blurred, as far as the colour of the button
        is see-through — give ``color`` an alpha below 1, ``(1, 1, 1, 0.5)``.

        A button that is "on" — the page shown, a switch — is the same call
        with the accent as ``color`` and black as ``text_color``.

        ``True`` comes back in exactly one frame per click::

            if ui.draw_button((20, 20), (120, 34), "Next"):
                page += 1

        ``enabled=False`` draws it faded and deaf: no hover, never ``True``.
        """
        pointer = self.pointer
        over = enabled and pointer.over(anchor, size, radius)
        if not enabled:  # faded, on a light background as on a dark one
            r, g, b, a = screen2d.rgba(color)
            body = (r, g, b, 0.45 * a)
        elif over:
            body = shade(hover_color, 0.8) if pointer.press is not None else hover_color
        else:
            body = color
        if blur > 0.0:
            self.draw_screen_acrylic_rect(anchor, size, body, radius, blur)
        else:
            self.draw_screen_rect(anchor, size, body, radius)
        if border_color is not None:
            self.draw_screen_rect_outline(anchor, size, border_color, radius, 1.0)

        write = getattr(self, "draw_screen_text", None)  # the plain renderer has no text
        if write is not None and name:
            width, height = abs(float(size[0])), abs(float(size[1]))
            shown = fit_text(self.measure_screen_text, name, text_size, width - 8.0)
            x = float(anchor[0]) + 0.5 * (float(size[0]) - self.measure_screen_text(shown, text_size))
            y = float(anchor[1]) + 0.5 * float(size[1]) + 0.5 * text_size  # the baseline
            if not enabled:  # the name, halfway to the colour of the button
                ink = tuple(0.5 * (t + c) for t, c in zip(text_color[:3], body[:3]))
            else:
                ink = hover_text_color if over else text_color
            write(x, y, shown, text_size, tuple(ink[:3]))

        clicked = enabled and pointer.clicked(anchor, size, radius)
        if clicked:
            # Whatever the program changes because of this click may concern
            # things already drawn in this frame: one more frame shows it.
            pointer.redraw()
        return clicked

    # ---------------------------------------------------------------- table
    def draw_table(
        self,
        anchor: screen2d.Point2,
        size: screen2d.Size2,
        rows: Sequence,
        text_color: screen2d.ColorRGBA = TABLE_TEXT,
        backgrounds: Sequence[screen2d.ColorRGBA] = TABLE_ROWS,
        *,
        header: Sequence[str] | None = None,
        header_color: screen2d.ColorRGBA = TABLE_HEADER_TEXT,
        header_background: screen2d.ColorRGBA = TABLE_HEADER,
        columns: Sequence[float] | None = None,
        align: str | Sequence[str] = "left",
        row_height: float = 26.0,
        text_size: float = 11.0,
        first: int = 0,
        hover_color: screen2d.ColorRGBA | None = None,
        selected: int | None = None,
        selected_color: screen2d.ColorRGBA = BUTTON_HOVER,
        blur: float = 0.0,
    ) -> int | None:
        """Draw a list as a table; return the line under the cursor (or ``None``).

        ``rows`` is the list: one entry per line. An entry is a string (one
        column) or a list of things (one per column) — numbers are written
        with ``str``. ``anchor`` and ``size`` are the place of the table.

        ``text_color`` is one colour, or a list with one colour per line.
        ``backgrounds`` is a list of colours used in turn, line after line:
        two colours make the classic striped table, where the eye follows a
        line without slipping to its neighbour.

        Options: ``header`` — the titles of the columns, on a line of their
        own (``header_color``, ``header_background``); ``columns`` — the
        relative widths (``[3, 1, 1]``: the first is three times wider),
        equal when not given; ``align`` — ``"left"``, ``"right"`` or
        ``"center"``, one for all or a list with one per column;
        ``row_height`` and ``text_size`` in pixels.

        Only the lines that fit in ``size`` are drawn, starting at line
        ``first`` — change ``first`` to scroll; a thin bar on the right (of the
        colour of the text) shows where you are. ``hover_color`` lights the line under the cursor and
        ``selected`` (a line number) marks one line with ``selected_color``.

        ``blur`` (pixels) puts frosted glass behind the table: what was drawn
        before it shows through, blurred, wherever a background is
        see-through. Give the backgrounds an alpha below 1 —
        ``[(1, 1, 1, 0.5), (0.9, 0.92, 0.95, 0.5)]`` — or nothing is seen of it.

        The number returned counts from the top of ``rows`` (not of what is
        visible), so ``rows[number]`` is the line pointed at.

        Cost: three draw calls whatever the size of the list (one more with ``blur``). The picture of
        the table is built once and kept until what it shows changes.
        """
        x0, y0 = float(anchor[0]), float(anchor[1])
        width, height = float(size[0]), float(size[1])
        row_height = max(float(row_height), 1.0)
        head = row_height if header is not None else 0.0
        visible = max(int((height - head) // row_height), 0)
        first = min(max(int(first), 0), max(len(rows) - visible, 0))
        shown = rows[first : first + visible]
        top = y0 + head

        # --- the geometry: built once, then looked up -------------------
        cells = tuple((str(row),) if isinstance(row, str) or not hasattr(row, "__iter__") else tuple(map(str, row)) for row in shown)
        key = (
            x0, y0, width, height, cells, first, len(rows), row_height, float(text_size),
            _frozen(text_color), _frozen(backgrounds), None if header is None else tuple(header),
            _frozen(header_color), _frozen(header_background), None if columns is None else tuple(columns),
            align if isinstance(align, str) else tuple(align),
        )  # fmt: skip
        cache = self.__dict__.setdefault("_table_cache", {})
        built = cache.pop(key, None)
        if built is None:
            built = self._build_table(
                x0, y0, width, cells, first, len(rows), visible, row_height, float(text_size),
                text_color, backgrounds, header, header_color, header_background, columns, align,
            )  # fmt: skip
            if len(cache) >= 32:  # forget the table used longest ago
                cache.pop(next(iter(cache)))
        cache[key] = built
        rects, strokes = built

        # --- the line under the cursor ----------------------------------
        hovered = None
        body = ((x0, top), (width, len(shown) * row_height))
        if shown and self.pointer.over(*body):
            hovered = first + min(int((self.pointer.cursor[1] - top) // row_height), len(shown) - 1)

        if blur > 0.0 and (shown or header is not None):
            # The glass carries no colour of its own (alpha 0): the stripes drawn
            # on it are the tint. Its rgb only serves when blur is switched off.
            glass = (*color_list(backgrounds, 1)[0][:3], 0.0)
            self.draw_screen_acrylic_rect((x0, y0), (width, head + len(shown) * row_height), glass, 0.0, blur)
        self._chart_rects(rects, sharp=True)
        marks = []
        if selected is not None and first <= selected < first + len(shown):
            marks.append(screen2d.rect_instance((x0, top + (selected - first) * row_height), (4.0, row_height), selected_color))
            marks.append(screen2d.rect_instance((x0, top + (selected - first) * row_height), (width, row_height), (*screen2d.rgba(selected_color)[:3], 0.18)))
        if hovered is not None and hover_color is not None:
            marks.append(screen2d.rect_instance((x0, top + (hovered - first) * row_height), (width, row_height), hover_color))
        if marks:
            self._chart_rects(np.concatenate(marks), sharp=True)
        if strokes is not None:
            self._submit_screen_strokes(strokes)
        return hovered

    def _build_table(
        self, x0, y0, width, cells, first, total, visible, row_height, text_size,
        text_color, backgrounds, header, header_color, header_background, columns, align,
    ):  # fmt: skip
        """The rectangles and the pen strokes of a table, as two arrays."""
        count = max([len(row) for row in cells] + [len(header) if header is not None else 0] + [1])
        weights = np.ones(count) if columns is None else np.resize(np.asarray(columns, dtype=np.float64), count)
        edges = x0 + width * np.concatenate(([0.0], np.cumsum(weights) / max(weights.sum(), 1e-9)))
        aligns = [align] * count if isinstance(align, str) else [align[i % len(align)] for i in range(count)]
        top = y0 + (row_height if header is not None else 0.0)
        pad = 10.0

        rects = np.zeros((len(cells) + 2, screen2d.SHAPE_FLOATS), dtype=np.float32)
        used = len(cells)
        rects[:used, 0] = x0 + 0.5 * width
        rects[:used, 1] = top + (np.arange(used) + 0.5) * row_height
        rects[:used, 2] = 0.5 * width
        rects[:used, 3] = 0.5 * row_height
        # The stripes belong to the lines, not to the screen: line 7 keeps its
        # colour while the table scrolls.
        rects[:used, 4:8] = color_list(backgrounds, first + used)[first:]
        if header is not None:
            rects[used] = screen2d.rect_instance((x0, y0), (width, row_height), header_background)
            used += 1
        if total > visible > 0:  # more lines than room: a bar shows where we are
            track = visible * row_height
            rects[used] = screen2d.rect_instance(
                (x0 + width - 4.0, top + track * first / total), (3.0, max(track * visible / total, 6.0)), (*color_list(text_color, 1)[0][:3], 0.35)
            )
            used += 1

        strokes = []
        text_strokes = getattr(self, "_screen_text_strokes", None)  # the plain renderer has no text
        if text_strokes is not None:
            inks = color_list(text_color, first + len(cells))[first:]
            lines = [(top + (index + 0.5) * row_height, row, inks[index][:3]) for index, row in enumerate(cells)]
            if header is not None:
                lines.append((y0 + 0.5 * row_height, tuple(header), screen2d.rgba(header_color)[:3]))
            for middle, row, ink in lines:
                for column, text in enumerate(row[:count]):
                    room = edges[column + 1] - edges[column] - 2.0 * pad
                    text = fit_text(self.measure_screen_text, text, text_size, room)
                    if not text:
                        continue
                    free = room - self.measure_screen_text(text, text_size)
                    shift = {"right": free, "center": 0.5 * free}.get(aligns[column], 0.0)
                    placed = text_strokes(edges[column] + pad + shift, middle + 0.5 * text_size, text, text_size, ink)
                    if placed is not None:
                        strokes.append(placed)
        return rects[:used], (np.concatenate(strokes) if strokes else None)


def _frozen(value) -> bytes:
    """One colour or a list of colours, as something a dictionary can use as a key."""
    count = 1 if np.ndim(value[0]) == 0 else len(value)
    return color_list(value, count).tobytes()
