"""Data widgets: values turned into pictures (mixed into both renderers).

Every widget is one method. It takes the place it may use — ``anchor`` (its
top-left corner) and ``size`` (width, height), in window pixels — then the
data, then the colours::

    ui.draw_pie((20, 20), (200, 200), [30, 50, 20], [RED, GREEN, BLUE])
    ui.draw_gauge((240, 20), (200, 200), pressure, 0, 10)
    ui.draw_line_chart((20, 240), (420, 120), history.values, CYAN)

They keep nothing between frames: call them from ``draw_frame`` with the
current values, and the picture follows the data. That makes them right for
live readings — a pressure, a wind speed, a water level, the torque of a motor.

Colours follow the rule of the gradient methods. Where a widget shows several
things (the slices of a pie, the bars of a graph), ``colors`` is a list with
one colour per thing; a shorter list is reused in turn. Where a widget shows
one value on a scale (a gauge, a level, a meter), ``stops`` and ``colors``
describe a gradient along that scale — ``[0, 0.7, 1], [GREEN, YELLOW, RED]``
turns red near the top.

Speed: a widget is a handful of draw calls whatever the amount of data. The
slices of a pie are one call, the bars of a graph one call, a curve of ten
thousand points one call. Text is the expensive part; every widget draws its
numbers only when asked (``label=...``).

Three widgets live in the 3D scene, lit and casting shadows like any solid:
``draw_bars_3d``, ``draw_pie_3d`` and ``draw_tank_3d``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import gpu3d
import numpy as np
import screen2d
import shaders3d
import solid3d

# Colours used when none are given: eight that stay distinct side by side.
PALETTE = (
    (0.20, 0.60, 1.00, 1.0),
    (1.00, 0.55, 0.10, 1.0),
    (0.30, 0.80, 0.40, 1.0),
    (0.90, 0.25, 0.35, 1.0),
    (0.65, 0.40, 0.95, 1.0),
    (1.00, 0.85, 0.20, 1.0),
    (0.20, 0.85, 0.85, 1.0),
    (0.95, 0.50, 0.75, 1.0),
)
TRACK = (0.30, 0.31, 0.36, 0.55)  # the unlit part of a gauge, a level, a meter
INK = (0.86, 0.88, 0.93, 1.0)  # needles, text
FAINT = (0.55, 0.57, 0.64, 0.55)  # grid lines, ticks
# A scale that reads "fine, careful, too much": the default of gauges and meters.
TRAFFIC_STOPS = (0.0, 0.6, 0.85, 1.0)
TRAFFIC_COLORS = ((0.25, 0.80, 0.40, 1.0), (0.95, 0.85, 0.20, 1.0), (1.00, 0.55, 0.10, 1.0), (0.95, 0.20, 0.25, 1.0))


class History:
    """The last ``size`` readings of a sensor, ready for ``draw_line_chart``.

    ``add(value)`` each time a reading arrives; ``values`` is always the
    readings in order, oldest first, as an array. Adding costs the same
    whatever the size: nothing is shifted or copied.

        speed = History(600)            # ten seconds at 60 readings a second
        speed.add(sensor.read())        # in draw_frame, or when data arrives
        ui.draw_line_chart(anchor, size, speed.values, CYAN)
    """

    def __init__(self, size: int = 256, fill: float = 0.0) -> None:
        self.size = max(int(size), 2)
        # Every reading is written twice, `size` apart, so the last `size`
        # readings always sit side by side somewhere in this array.
        self._data = np.full(2 * self.size, float(fill), dtype=np.float32)
        self._next = 0
        self.count = 0  # readings received so far (stops growing at `size`)

    def add(self, value: float) -> None:
        self._data[self._next] = self._data[self._next + self.size] = value
        self._next = (self._next + 1) % self.size
        self.count = min(self.count + 1, self.size)

    @property
    def values(self) -> np.ndarray:
        """The readings, oldest first (a view: do not keep it across an ``add``)."""
        return self._data[self._next : self._next + self.size]

    @property
    def last(self) -> float:
        """The most recent reading."""
        return float(self._data[(self._next - 1) % self.size])


def color_list(colors, count: int) -> np.ndarray:
    """``count`` rows of ``(r, g, b, a)`` from one colour, a list of colours or nothing."""
    if colors is None:
        colors = PALETTE
    array = np.asarray(colors, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.shape[1] == 3:
        array = np.concatenate((array, np.ones((len(array), 1), dtype=np.float32)), axis=1)
    return array[np.arange(count) % len(array)]


def ramp(stops, colors, t) -> np.ndarray:
    """The colours of a gradient at many positions at once: ``(len(t), 4)``."""
    positions, values = screen2d.gradient_stops(stops, colors)
    table = np.asarray(values, dtype=np.float64)
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return np.stack([np.interp(t, positions, table[:, c]) for c in range(4)], axis=-1).astype(np.float32)


def fraction(value, minimum: float, maximum: float):
    """Where ``value`` lies between two ends, from 0 to 1 (clamped)."""
    span = float(maximum) - float(minimum)
    if span == 0.0:
        return np.zeros_like(np.asarray(value, dtype=np.float64)) if np.ndim(value) else 0.0
    return np.clip((np.asarray(value, dtype=np.float64) - minimum) / span, 0.0, 1.0)


def clock(degrees) -> np.ndarray:
    """Clock degrees (0 = up, clockwise) -> screen radians (0 = right, clockwise)."""
    return np.radians(np.asarray(degrees, dtype=np.float64) - 90.0)


class ChartMethods:
    """``draw_pie``, ``draw_gauge`` and the rest. Needs the screen pipelines of the renderer."""

    # ------------------------------------------------------------- batches
    # Each of these draws any number of items with ONE draw call.
    def _chart_rects(self, rows: np.ndarray, sharp: bool = False) -> None:
        if not len(rows):
            return
        if sharp:  # exact edges, so neighbouring cells touch without a seam
            self._submit_points(self.sharp_rect_pipelines[True], self._screen_uniforms(0.0), rows)
        else:
            self._submit_points(self.shape_pipeline, self._screen_uniforms(1.0), rows)

    def _chart_lines(self, starts, ends, colors, thickness: float) -> None:
        starts = np.asarray(starts, dtype=np.float32).reshape(-1, 2)
        if not len(starts):
            return
        rows = np.zeros((len(starts), screen2d.LINE_FLOATS), dtype=np.float32)
        rows[:, 0:2] = starts
        rows[:, 2:4] = np.asarray(ends, dtype=np.float32).reshape(-1, 2)
        rows[:, 4:8] = colors
        rows[:, 8] = 0.5 * float(thickness)
        self._submit_points(self.screen_line_pipeline, self._screen_uniforms(1.0), rows)

    def _chart_triangles(self, corners: np.ndarray, colors) -> None:
        corners = np.asarray(corners, dtype=np.float32).reshape(-1, 6)
        if not len(corners):
            return
        rows = np.empty((len(corners), screen2d.TRIANGLE_FLOATS), dtype=np.float32)
        rows[:, 0:6] = corners
        rows[:, 6:10] = colors
        self.gpu.submit(
            gpu3d.DrawCall(
                pipeline=self.sharp_triangle_pipelines[True],
                uniforms=self._screen_uniforms(0.0),
                vertex_count=3,
                instance_count=len(rows),
                vertices=rows,
            )
        )

    def _chart_arcs(self, center, inner, outer, start, end, colors) -> None:
        """Slices of a ring; angles in clock degrees. Arrays or single values."""
        start = np.atleast_1d(np.asarray(start, dtype=np.float64))
        end = np.atleast_1d(np.asarray(end, dtype=np.float64))
        keep = end > start
        if not keep.any():
            return
        count = len(start)
        rows = np.zeros((count, 10), dtype=np.float32)
        rows[:, 0:2] = center
        rows[:, 2] = inner
        rows[:, 3] = outer
        rows[:, 4] = clock(0.5 * (start + end))
        rows[:, 5] = np.minimum(np.radians(0.5 * (end - start)), math.pi)
        rows[:, 6:10] = colors
        if self.__dict__.get("_arc_pipeline") is None:
            self._arc_pipeline = self.gpu.pipeline(
                shaders3d.SCREEN_ARCS,
                gpu3d.vertex_layout(*shaders3d.SCREEN_ARC_LAYOUT, instanced=True),
                blend=True,
                depth_write=False,
                depth_compare="always",
            )
        self._submit_points(self._arc_pipeline, self._screen_uniforms(1.0), rows[keep])

    def _chart_text(self, x: float, y: float, text: str, size: float, color, align: str = "left") -> None:
        """Text, if this renderer can write (the plain one cannot)."""
        write = getattr(self, "draw_screen_text", None)
        if write is None or not text:
            return
        if align != "left":
            width = self.measure_screen_text(text, size)
            x -= width if align == "right" else 0.5 * width
        write(x, y, text, size, tuple(color[:3]))

    def _chart_background(self, anchor, size, background) -> None:
        if background is not None:
            self.draw_screen_rect(anchor, size, background, min(10.0, 0.5 * min(size)))

    # ---------------------------------------------------------------- arcs
    def draw_screen_arc(self, center, radius: float, start: float, end: float, color, thickness: float = 0.0) -> None:
        """Draw an arc of a circle, or a slice of a disc, in screen space.

        Angles are in degrees like the hours of a clock: 0 is up, 90 to the
        right, and they grow clockwise. The arc goes from ``start`` to ``end``.
        ``thickness`` is the width of the line, drawn inward from ``radius``;
        0 (the default) fills the whole slice down to the centre, like a piece
        of pie. Anti-aliased, one draw call.
        """
        inner = max(float(radius) - float(thickness), 0.0) if thickness > 0.0 else 0.0
        self._chart_arcs(center, inner, float(radius), start, end, screen2d.rgba(color))

    # ----------------------------------------------------------------- pie
    def draw_pie(
        self, anchor, size, values, colors=None, *, hole: float = 0.0, start: float = 0.0, gap: float = 0.0,
        background=None,
    ) -> None:
        """A pie chart: each value is a slice, as large as its share of the total.

        ``colors`` gives one colour per slice. ``hole`` (0 to 1) empties the
        middle and makes it a ring ("donut") chart; ``start`` turns the first
        slice away from the top, in degrees; ``gap`` leaves that many degrees
        of space between slices. All the slices are one draw call.
        """
        self._chart_background(anchor, size, background)
        values = np.maximum(np.asarray(values, dtype=np.float64).ravel(), 0.0)
        total = float(values.sum())
        if total <= 0.0:
            return
        edges = start + 360.0 * np.concatenate(([0.0], np.cumsum(values) / total))
        outer = 0.5 * min(size) - 1.0
        center = (anchor[0] + 0.5 * size[0], anchor[1] + 0.5 * size[1])
        trim = 0.5 * gap if len(values) > 1 else 0.0
        self._chart_arcs(
            center, outer * min(max(hole, 0.0), 0.98), outer, edges[:-1] + trim, edges[1:] - trim,
            color_list(colors, len(values)),
        )

    # ---------------------------------------------------------------- bars
    def draw_bars(
        self, anchor, size, values, colors=None, *, minimum: float = 0.0, maximum: float | None = None,
        gap: float = 0.25, horizontal: bool = False, radius: float = 2.0, background=None,
    ) -> None:
        """A bar graph: one bar per value, its length the value.

        The scale goes from ``minimum`` to ``maximum`` (the largest value when
        not given — give it for live data, or the bars rescale as they move).
        Bars start from zero, so negative values hang the other way.
        ``colors`` is one colour for all or one per bar; ``gap`` is the share
        of each slot left empty; ``horizontal`` lays the bars on their side.
        All the bars are one draw call.
        """
        self._chart_background(anchor, size, background)
        values = np.asarray(values, dtype=np.float64).ravel()
        count = len(values)
        if not count:
            return
        top = float(values.max()) if maximum is None else float(maximum)
        top = top if top != minimum else minimum + 1.0
        zero = float(fraction(0.0, minimum, top))  # where the bars start from
        ends = fraction(values, minimum, top)
        low, high = np.minimum(ends, zero), np.maximum(ends, zero)

        along, across = (size[1], size[0]) if horizontal else (size[0], size[1])
        slot = along / count
        width = slot * (1.0 - min(max(gap, 0.0), 0.95))
        middle = (np.arange(count) + 0.5) * slot
        rows = np.zeros((count, screen2d.SHAPE_FLOATS), dtype=np.float32)
        if horizontal:
            rows[:, 0] = anchor[0] + 0.5 * (low + high) * across
            rows[:, 1] = anchor[1] + middle
            rows[:, 2] = 0.5 * (high - low) * across
            rows[:, 3] = 0.5 * width
        else:
            rows[:, 0] = anchor[0] + middle
            rows[:, 1] = anchor[1] + (1.0 - 0.5 * (low + high)) * across
            rows[:, 2] = 0.5 * width
            rows[:, 3] = 0.5 * (high - low) * across
        rows[:, 4:8] = color_list(colors if colors is not None else PALETTE[0], count)
        rows[:, 8] = np.minimum(radius, np.minimum(rows[:, 2], rows[:, 3]))
        self._chart_rects(rows)

    def draw_stacked_bar(self, anchor, size, values, colors=None, *, horizontal: bool | None = None, background=None) -> None:
        """One bar cut into parts, each as long as its share of the total.

        The flat cousin of the pie chart: how a whole divides (a disk, a
        budget, the time spent in each state). It runs along the longer side
        unless ``horizontal`` says otherwise. One draw call.
        """
        self._chart_background(anchor, size, background)
        values = np.maximum(np.asarray(values, dtype=np.float64).ravel(), 0.0)
        total = float(values.sum())
        if total <= 0.0:
            return
        if horizontal is None:
            horizontal = size[0] >= size[1]
        edges = np.concatenate(([0.0], np.cumsum(values) / total))
        rows = np.zeros((len(values), screen2d.SHAPE_FLOATS), dtype=np.float32)
        middle, half = 0.5 * (edges[:-1] + edges[1:]), 0.5 * (edges[1:] - edges[:-1])
        if horizontal:
            rows[:, 0], rows[:, 1] = anchor[0] + middle * size[0], anchor[1] + 0.5 * size[1]
            rows[:, 2], rows[:, 3] = half * size[0], 0.5 * size[1]
        else:
            rows[:, 0], rows[:, 1] = anchor[0] + 0.5 * size[0], anchor[1] + (1.0 - middle) * size[1]
            rows[:, 2], rows[:, 3] = 0.5 * size[0], half * size[1]
        rows[:, 4:8] = color_list(colors, len(values))
        self._chart_rects(rows, sharp=True)

    # ---------------------------------------------------------- line chart
    def draw_line_chart(
        self, anchor, size, values, colors=None, *, minimum: float | None = None, maximum: float | None = None,
        thickness: float = 2.0, fill: float = 0.0, grid: int = 0, background=None,
    ) -> None:
        """A curve through a series of values: how something changed over time.

        ``values`` is the series, oldest first — a list, an array, or
        ``History.values``. A 2D array (one row per series) draws several
        curves, with one colour each from ``colors``. The vertical scale goes
        from ``minimum`` to ``maximum``; left out, it follows the data.
        ``fill`` (0 to 1) shades the area under each curve with that opacity;
        ``grid`` draws that many horizontal guide lines.

        One draw call per curve whatever its length, plus one for its fill.
        """
        self._chart_background(anchor, size, background)
        series = np.atleast_2d(np.asarray(values, dtype=np.float32))
        points = series.shape[1]
        if points < 2:
            return
        low = float(series.min()) if minimum is None else float(minimum)
        high = float(series.max()) if maximum is None else float(maximum)
        if high <= low:
            low, high = low - 0.5, low + 0.5
        if grid > 0:
            ys = anchor[1] + size[1] * np.linspace(0.0, 1.0, grid + 2)[1:-1]
            starts = np.stack((np.full(grid, anchor[0]), ys), axis=1)
            self._chart_lines(starts, starts + (size[0], 0.0), FAINT, 1.0)
        xs = anchor[0] + size[0] * np.linspace(0.0, 1.0, points, dtype=np.float32)
        palette = color_list(colors if colors is not None else PALETTE, len(series))
        bottom = anchor[1] + size[1]
        for row, color in zip(series, palette):
            ys = bottom - size[1] * fraction(row, low, high).astype(np.float32)
            curve = np.stack((xs, ys), axis=1)
            if fill > 0.0:
                # The area under each segment is a four-sided strip: two triangles.
                a, b = curve[:-1], curve[1:]
                floor_a = np.stack((a[:, 0], np.full(len(a), bottom, dtype=np.float32)), axis=1)
                floor_b = np.stack((b[:, 0], np.full(len(b), bottom, dtype=np.float32)), axis=1)
                strips = np.concatenate(
                    (np.concatenate((a, b, floor_b), axis=1), np.concatenate((a, floor_b, floor_a), axis=1))
                )
                self._chart_triangles(strips, (color[0], color[1], color[2], color[3] * fill))
            self.draw_screen_polyline(curve, color, thickness)

    def draw_scatter(
        self, anchor, size, xs, ys, colors=None, *, x_range=None, y_range=None, point: float = 3.0, background=None
    ) -> None:
        """A cloud of points: one dot per ``(x, y)`` pair, to see how two things relate.

        ``x_range`` and ``y_range`` are ``(minimum, maximum)``; left out, they
        follow the data. ``colors`` is one colour or one per point; ``point``
        the radius of a dot in pixels. All the dots are one draw call.
        """
        self._chart_background(anchor, size, background)
        xs = np.asarray(xs, dtype=np.float64).ravel()
        ys = np.asarray(ys, dtype=np.float64).ravel()
        count = min(len(xs), len(ys))
        if not count:
            return
        x_low, x_high = x_range if x_range is not None else (xs.min(), xs.max())
        y_low, y_high = y_range if y_range is not None else (ys.min(), ys.max())
        rows = np.zeros((count, screen2d.SHAPE_FLOATS), dtype=np.float32)
        rows[:, 0] = anchor[0] + size[0] * fraction(xs[:count], x_low, x_high)
        rows[:, 1] = anchor[1] + size[1] * (1.0 - fraction(ys[:count], y_low, y_high))
        rows[:, 2:4] = point
        rows[:, 4:8] = color_list(colors if colors is not None else PALETTE[0], count)
        rows[:, 8] = point
        self._chart_rects(rows)

    def draw_heatmap(
        self, anchor, size, grid, stops=(0.0, 0.5, 1.0), colors=((0.1, 0.2, 0.8), (0.9, 0.9, 0.2), (0.95, 0.2, 0.2)),
        *, minimum: float | None = None, maximum: float | None = None, gap: float = 0.0, background=None,
    ) -> None:
        """A table of values as coloured cells: a grid of sensors at a glance.

        ``grid`` is a 2D array (rows, columns). Each cell takes the colour of
        the gradient ``stops`` / ``colors`` at its value, on a scale from
        ``minimum`` to ``maximum`` (the data's own range when left out).
        ``gap`` leaves that many pixels between cells. One draw call.
        """
        self._chart_background(anchor, size, background)
        grid = np.atleast_2d(np.asarray(grid, dtype=np.float64))
        rows_n, cols_n = grid.shape
        if not grid.size:
            return
        low = float(grid.min()) if minimum is None else float(minimum)
        high = float(grid.max()) if maximum is None else float(maximum)
        cell_w, cell_h = size[0] / cols_n, size[1] / rows_n
        col, row = np.meshgrid(np.arange(cols_n), np.arange(rows_n))
        rows = np.zeros((grid.size, screen2d.SHAPE_FLOATS), dtype=np.float32)
        rows[:, 0] = anchor[0] + (col.ravel() + 0.5) * cell_w
        rows[:, 1] = anchor[1] + (row.ravel() + 0.5) * cell_h
        rows[:, 2] = max(0.5 * cell_w - 0.5 * gap, 0.5)
        rows[:, 3] = max(0.5 * cell_h - 0.5 * gap, 0.5)
        rows[:, 4:8] = ramp(stops, colors, fraction(grid.ravel(), low, high))
        self._chart_rects(rows, sharp=True)

    def draw_radar(
        self, anchor, size, values, color=None, *, maximum: float | None = None, rings: int = 4, fill: float = 0.3,
        thickness: float = 2.0, background=None,
    ) -> None:
        """A radar (spider) chart: one spoke per value, joined into a shape.

        Good for comparing several readings of the same kind at once — the
        torque of six motors, the load of each leg. The scale goes from the
        centre (0) to the rim (``maximum``, the largest value when left out).
        ``rings`` is the number of guide rings, ``fill`` the opacity of the
        shape's inside.
        """
        self._chart_background(anchor, size, background)
        values = np.maximum(np.asarray(values, dtype=np.float64).ravel(), 0.0)
        count = len(values)
        if count < 3:
            return
        top = float(values.max()) if maximum is None else float(maximum)
        center = np.array((anchor[0] + 0.5 * size[0], anchor[1] + 0.5 * size[1]))
        radius = 0.5 * min(size) - 2.0
        angle = clock(np.arange(count) * 360.0 / count)
        spokes = np.stack((np.cos(angle), np.sin(angle)), axis=1)
        # The web: the spokes, then the rings — every line in one draw call.
        starts = [np.repeat(center[None, :], count, axis=0)]
        ends = [center + spokes * radius]
        for ring in range(1, rings + 1):
            corners = center + spokes * (radius * ring / rings)
            starts.append(corners)
            ends.append(np.roll(corners, -1, axis=0))
        self._chart_lines(np.concatenate(starts), np.concatenate(ends), FAINT, 1.0)
        r, g, b, a = screen2d.rgba(color if color is not None else PALETTE[0])
        shape = center + spokes * (radius * fraction(values, 0.0, top if top > 0 else 1.0))[:, None]
        if fill > 0.0:
            fan = np.concatenate(
                (np.repeat(center[None, :], count, axis=0), shape, np.roll(shape, -1, axis=0)), axis=1
            )
            self._chart_triangles(fan, (r, g, b, a * fill))
        self.draw_screen_polyline(shape, (r, g, b, a), thickness, closed=True)

    # ------------------------------------------------------ one value on a scale
    def draw_gauge(
        self, anchor, size, value: float, minimum: float = 0.0, maximum: float = 1.0, *,
        stops=TRAFFIC_STOPS, colors=TRAFFIC_COLORS, track=TRACK, needle=INK, thickness: float = 0.16,
        sweep: float = 270.0, label: str | None = None, background=None,
    ) -> None:
        """A dial with a needle: one value on a round scale, like a pressure gauge.

        The scale runs from ``minimum`` to ``maximum`` around an arc of
        ``sweep`` degrees (270 by default, open at the bottom). The part of
        the arc up to the value is coloured by the gradient ``stops`` /
        ``colors`` (position 0 is ``minimum``, 1 is ``maximum``); the rest
        shows ``track``. ``thickness`` is the width of the arc as a share of
        the radius; ``needle=None`` removes the needle.

        ``label`` writes a text in the middle: pass the value formatted the
        way you want it, for instance ``f"{pressure:.1f} bar"``.
        """
        self._chart_background(anchor, size, background)
        center = (anchor[0] + 0.5 * size[0], anchor[1] + 0.5 * size[1])
        outer = 0.5 * min(size) - 1.0
        inner = outer * (1.0 - min(max(thickness, 0.02), 1.0))
        first = -0.5 * sweep
        share = float(fraction(value, minimum, maximum))
        self._chart_arcs(center, inner, outer, first, first + sweep, screen2d.rgba(track))
        if share > 0.0:
            # The coloured part is cut into short pieces, each with the colour
            # of the gradient at its place: one draw call for all of them.
            pieces = max(int(48 * share), 1)
            edges = first + sweep * share * np.linspace(0.0, 1.0, pieces + 1)
            middle = 0.5 * (edges[:-1] + edges[1:])
            self._chart_arcs(
                center, inner, outer, edges[:-1], edges[1:] + 0.3,
                ramp(stops, colors, (middle - first) / sweep),
            )
        if needle is not None:
            angle = float(clock(first + sweep * share))
            tip = (center[0] + math.cos(angle) * outer * 0.92, center[1] + math.sin(angle) * outer * 0.92)
            self.draw_screen_line(center, tip, needle, max(outer * 0.035, 1.5))
            self.draw_screen_circle(center, max(outer * 0.07, 2.5), needle)
        if label:
            text_size = max(outer * 0.2, 9.0)
            self._chart_text(center[0], center[1] + outer * 0.62, label, text_size, INK, "center")

    def draw_level(
        self, anchor, size, value: float, minimum: float = 0.0, maximum: float = 1.0, color=None, *,
        stops=None, colors=None, track=TRACK, radius: float = 6.0, horizontal: bool | None = None,
        label: str | None = None,
    ) -> None:
        """A container that fills up: a water level, a battery, a progress bar.

        The filled part grows from the bottom (or from the left when the
        widget is wider than tall, or ``horizontal=True``) as ``value`` goes
        from ``minimum`` to ``maximum``. It has one ``color``, or — given
        ``stops`` and ``colors`` — the colour of that gradient at the value,
        so it can turn red when nearly empty or nearly full.
        """
        if horizontal is None:
            horizontal = size[0] > size[1]
        share = float(fraction(value, minimum, maximum))
        if stops is not None and colors is not None:
            color = tuple(ramp(stops, colors, [share])[0])
        elif color is None:
            color = PALETTE[0]
        radius = min(radius, 0.5 * min(size))
        self.draw_screen_rect(anchor, size, track, radius)
        if share > 0.0:
            if horizontal:
                filled = (max(size[0] * share, 2.0 * radius), size[1])
                self.draw_screen_rect(anchor, filled, color, radius)
            else:
                height = max(size[1] * share, 2.0 * radius)
                self.draw_screen_rect((anchor[0], anchor[1] + size[1] - height), (size[0], height), color, radius)
        if label:
            text_size = max(min(0.22 * min(size), 16.0), 9.0)
            self._chart_text(
                anchor[0] + 0.5 * size[0], anchor[1] + 0.5 * size[1] + 0.4 * text_size, label, text_size, INK, "center"
            )

    def draw_meter(
        self, anchor, size, value: float, minimum: float = 0.0, maximum: float = 1.0, *, segments: int = 12,
        stops=TRAFFIC_STOPS, colors=TRAFFIC_COLORS, track=TRACK, gap: float = 2.0, horizontal: bool | None = None,
    ) -> None:
        """A row of lights that come on one after the other: a signal or sound level.

        ``segments`` lights share the widget; as many as the value deserves
        are lit, each in the colour of the gradient ``stops`` / ``colors`` at
        its own place — so the top ones can be red whatever the value. It
        runs along the longer side. One draw call.
        """
        if horizontal is None:
            horizontal = size[0] > size[1]
        segments = max(int(segments), 1)
        lit = int(round(float(fraction(value, minimum, maximum)) * segments))
        place = (np.arange(segments) + 0.5) / segments
        rows = np.zeros((segments, screen2d.SHAPE_FLOATS), dtype=np.float32)
        along = size[0] if horizontal else size[1]
        cell = along / segments
        half = max(0.5 * (cell - gap), 0.5)
        if horizontal:
            rows[:, 0], rows[:, 1] = anchor[0] + place * size[0], anchor[1] + 0.5 * size[1]
            rows[:, 2], rows[:, 3] = half, 0.5 * size[1]
        else:
            rows[:, 0], rows[:, 1] = anchor[0] + 0.5 * size[0], anchor[1] + (1.0 - place) * size[1]
            rows[:, 2], rows[:, 3] = 0.5 * size[0], half
        rows[:, 4:8] = screen2d.rgba(track)
        rows[:lit, 4:8] = ramp(stops, colors, place[:lit])
        rows[:, 8] = np.minimum(2.0, np.minimum(rows[:, 2], rows[:, 3]))
        self._chart_rects(rows)

    def draw_compass(
        self, anchor, size, heading: float, color=None, *, track=TRACK, ticks: int = 16, labels: bool = True,
        background=None,
    ) -> None:
        """A compass rose with a needle: a direction — the wind, a heading, a bearing.

        ``heading`` is in degrees: 0 points up (north), 90 to the right
        (east). ``ticks`` marks are spread around the rim; ``labels`` writes
        N, E, S and W.
        """
        self._chart_background(anchor, size, background)
        center = np.array((anchor[0] + 0.5 * size[0], anchor[1] + 0.5 * size[1]))
        outer = 0.5 * min(size) - 1.0
        self._chart_arcs(center, outer * 0.94, outer, 0.0, 360.0, screen2d.rgba(track))
        if ticks > 0:
            angle = clock(np.arange(ticks) * 360.0 / ticks)
            out = np.stack((np.cos(angle), np.sin(angle)), axis=1)
            long = (np.arange(ticks) % max(ticks // 4, 1)) == 0  # the four main directions
            self._chart_lines(center + out * outer * 0.9, center + out * (outer * np.where(long, 0.74, 0.82))[:, None], FAINT, 1.5)
        if labels:
            text_size = max(outer * 0.17, 8.0)
            for letter, (dx, dy) in zip("NESW", ((0, -1), (1, 0), (0, 1), (-1, 0))):
                self._chart_text(
                    center[0] + dx * outer * 0.58, center[1] + dy * outer * 0.58 + 0.4 * text_size, letter, text_size,
                    INK, "center",
                )
        # The needle: a long pointed half toward the heading, a short dim tail.
        r, g, b, a = screen2d.rgba(color if color is not None else PALETTE[3])
        angle = float(clock(heading))
        ahead = np.array((math.cos(angle), math.sin(angle)))
        side = np.array((-ahead[1], ahead[0])) * outer * 0.09
        tip, tail = center + ahead * outer * 0.7, center - ahead * outer * 0.38
        self._chart_triangles(
            np.array([(*(center - side), *(center + side), *tip), (*(center - side), *(center + side), *tail)]),
            np.array([(r, g, b, a), (r * 0.45, g * 0.45, b * 0.45, a)], dtype=np.float32),
        )
        self.draw_screen_circle(center, max(outer * 0.05, 2.0), INK)

    # -------------------------------------------------------------- in 3D
    def draw_bars_3d(
        self, origin, size, values, colors=None, *, maximum: float | None = None, gap: float = 0.25, **material
    ) -> None:
        """A bar graph standing in the 3D scene: lit blocks that cast shadows.

        ``origin`` is the corner of the graph on the floor, ``(x, y, z)``;
        ``size`` is ``(length, depth, height)``: the bars line up along x,
        are ``depth`` thick along y, and the largest is ``height`` tall.
        ``values`` may be a 2D array (rows along y) for a field of bars.
        ``colors`` is one colour, or one per bar of a row. Extra named
        arguments (``shader=...``) go to the solids.
        """
        grid = np.atleast_2d(np.asarray(values, dtype=np.float64))
        rows_n, cols_n = grid.shape
        if not grid.size:
            return
        top = float(grid.max()) if maximum is None else float(maximum)
        if top <= 0.0:
            return
        slot_x, slot_y = size[0] / cols_n, size[1] / rows_n
        wide = min(max(1.0 - gap, 0.05), 1.0)
        palette = color_list(colors, cols_n)
        for row in range(rows_n):
            for col in range(cols_n):
                height = size[2] * float(fraction(grid[row, col], 0.0, top))
                if height <= 1e-4:
                    continue
                center = (
                    origin[0] + (col + 0.5) * slot_x,
                    origin[1] + (row + 0.5) * slot_y,
                    origin[2] + 0.5 * height,
                )
                self.draw_box(center, (slot_x * wide, slot_y * wide, height), tuple(palette[col]), **material)

    def draw_pie_3d(
        self, center, radius: float, height: float, values, colors=None, *, start: float = 0.0, explode: float = 0.0,
        **material,
    ) -> None:
        """A pie chart with thickness, lying in the 3D scene.

        ``center`` is the middle of its underside, ``(x, y, z)``. Each value
        is a slice as large as its share of the total; ``explode`` pushes
        every slice outward by that distance, to separate them. Slices are
        cut on whole degrees. Extra named arguments go to the solids.
        """
        values = np.maximum(np.asarray(values, dtype=np.float64).ravel(), 0.0)
        total = float(values.sum())
        if total <= 0.0:
            return
        edges = np.round(360.0 * np.concatenate(([0.0], np.cumsum(values) / total))).astype(int)
        palette = color_list(colors, len(values))
        wedges = self.__dict__.setdefault("_chart_wedges", {})
        for index in range(len(values)):
            degrees = int(edges[index + 1] - edges[index])
            if degrees <= 0:
                continue
            mesh = wedges.get(degrees)
            if mesh is None:  # a slice of this opening is built once, then reused
                mesh = wedges[degrees] = solid3d.Mesh(*wedge(degrees))
            turn = start + float(edges[index])
            middle = math.radians(90.0 - (turn + 0.5 * degrees))
            position = (
                center[0] + explode * math.cos(middle),
                center[1] + explode * math.sin(middle),
                center[2],
            )
            self.draw_mesh(
                mesh, tuple(palette[index]), position, (radius, radius, height), (0.0, 0.0, -turn), **material
            )

    def draw_tank_3d(
        self, base, radius: float, height: float, value: float, minimum: float = 0.0, maximum: float = 1.0,
        color=None, *, glass=(0.75, 0.88, 1.0, 0.16),
    ) -> None:
        """A glass tank with a liquid inside, in the 3D scene: a level you can walk around.

        ``base`` is the middle of its bottom, ``(x, y, z)``. The liquid rises
        from empty at ``minimum`` to full at ``maximum``. ``glass`` is the
        colour of the tank, see-through; ``None`` draws the liquid alone.
        Draw it after the opaque things of the scene, like every see-through
        solid.
        """
        share = float(fraction(value, minimum, maximum))
        x, y, z = (float(v) for v in base[:3])
        if share > 0.0:
            liquid = color if color is not None else PALETTE[0]
            self.draw_cylinder((x, y, z + 0.01), (x, y, z + max(height * share, 0.02)), radius * 0.9, liquid[:3])
        if glass is not None:
            self.draw_cylinder((x, y, z), (x, y, z + height), radius, glass, refraction=1.05)

    def _close_charts(self) -> None:
        for mesh in self.__dict__.pop("_chart_wedges", {}).values():
            mesh.destroy()


def wedge(degrees: int, step: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """A slice of a disc of radius 1 and height 1: triangles and their normals.

    It starts at "up" (+y) and opens clockwise seen from above, like the
    slices of the flat pie chart.
    """
    count = max(int(math.ceil(degrees / step)), 1)
    angle = np.radians(90.0 - np.linspace(0.0, degrees, count + 1))
    rim = np.stack((np.cos(angle), np.sin(angle)), axis=1)
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []

    def face(corners, normal_of) -> None:
        for corner in corners:
            positions.append(corner)
            normals.append(normal_of(corner))

    for a, b in zip(rim[:-1], rim[1:]):
        face([(0, 0, 1), (a[0], a[1], 1), (b[0], b[1], 1)], lambda _c: (0.0, 0.0, 1.0))  # top
        face([(0, 0, 0), (b[0], b[1], 0), (a[0], a[1], 0)], lambda _c: (0.0, 0.0, -1.0))  # bottom
        wall = [(a[0], a[1], 0), (b[0], b[1], 0), (b[0], b[1], 1), (a[0], a[1], 0), (b[0], b[1], 1), (a[0], a[1], 1)]
        face(wall, lambda c: (c[0], c[1], 0.0))  # the round side: the normal points outward
    if degrees < 360:
        for edge, sign in ((rim[0], 1.0), (rim[-1], -1.0)):  # the two flat cuts
            normal = (-edge[1] * sign, edge[0] * sign, 0.0)
            cut = [(0, 0, 0), (edge[0], edge[1], 0), (edge[0], edge[1], 1), (0, 0, 0), (edge[0], edge[1], 1), (0, 0, 1)]
            face(cut, lambda _c, n=normal: n)
    return np.asarray(positions, dtype=np.float32), np.asarray(normals, dtype=np.float32)
