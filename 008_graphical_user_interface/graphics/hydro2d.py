"""A journal for an indoor hydroponic culture: one year of readings, zoomable.

Two objects, made once::

    from hydro2d import GrowLog, GrowJournal

    log = GrowLog(sampling=60)            # one reading a minute, for this year
    journal = GrowJournal(log)

and two lines in ``draw_frame``::

    if log.due():                         # has the sampling delay passed?
        log.add(light=1, pump=0, water_temp=20.4, air_temp=24.1,
                humidity=61, ph=5.9, ec=1.8, level=72)
    journal.draw(app, (20, 20), (900, 560))

**The log** keeps what a grower watches: the light cycle and the watering
cycle (on / off), the temperature of the water and of the air, the humidity,
the pH and the strength (EC) of the nutrient solution, the level of the tank.
Each has a *safe range*; a reading outside it is an alert.

``sampling`` is the delay between two readings, in seconds. ``log.due()`` is
``True`` once per delay, so the sensors are read no more often than needed,
and two readings in the same delay overwrite each other instead of piling up.
A year at one reading a minute is half a million values per channel: 2 MB.

**The journal** shows the log at three zooms:

* *year* — one square per day, twelve lines of them: green where the chosen
  reading stayed in its safe range, amber where it touched a limit, blue or
  red where its average left the range. A bad week is seen from across the room;
* *month* — every channel as its own lane, hour by hour: the light and
  watering cycles as stripes, each reading as a curve inside the band of its
  hourly lowest and highest;
* *day* — the same lanes at the full sampling: when the lights went on, each
  watering, the curve of each reading, and the values under the cursor.

Click a month or a day to zoom in; the buttons at the top (year, month, day)
zoom back out; ``<`` and ``>`` go to the neighbour. The wheel or a pinch zooms
in and out at the cursor.

Drawing a year must not mean drawing half a million points: the log keeps,
beside the readings, their lowest / average / highest per hour and per day,
updated as readings arrive. The year view reads 366 numbers, the month view
744, the day view one day. Every lane is one or two draw calls.

``log.save(path)`` and ``GrowLog.load(path)`` keep the year between runs.
"""

from __future__ import annotations

import calendar
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import screen2d


@dataclass(frozen=True)
class Channel:
    """One thing that is logged.

    ``safe`` is the range it should stay in, ``scale`` the range drawn.
    ``cycle=True`` is something that is on or off (lights, a pump): log 1 or 0.
    """

    key: str
    name: str
    unit: str = ""
    safe: tuple[float, float] = (0.0, 1.0)
    scale: tuple[float, float] = (0.0, 1.0)
    color: tuple[float, float, float] = (0.2, 0.8, 0.8)
    cycle: bool = False
    decimals: int = 1


# What an indoor hydroponic system is usually watched for. Safe ranges are the
# common ones for leafy and fruiting crops; pass your own tuple of channels to
# GrowLog for another crop.
CHANNELS = (
    Channel("light", "Lights", cycle=True, color=(1.00, 0.82, 0.25)),
    Channel("pump", "Watering", cycle=True, color=(0.30, 0.62, 1.00)),
    Channel("water_temp", "Water temp", "°C", safe=(18.0, 22.0), scale=(14.0, 28.0), color=(0.20, 0.80, 0.85)),
    Channel("air_temp", "Air temp", "°C", safe=(20.0, 28.0), scale=(14.0, 34.0), color=(1.00, 0.60, 0.30)),
    Channel("humidity", "Humidity", "%", safe=(50.0, 70.0), scale=(30.0, 90.0), color=(0.55, 0.70, 1.00), decimals=0),
    Channel("ph", "pH", "", safe=(5.5, 6.5), scale=(4.5, 7.5), color=(0.75, 0.55, 1.00), decimals=2),
    Channel("ec", "EC", "mS", safe=(1.2, 2.4), scale=(0.6, 3.0), color=(0.45, 0.85, 0.50), decimals=2),
    Channel("level", "Tank level", "%", safe=(30.0, 100.0), scale=(0.0, 100.0), color=(0.35, 0.60, 0.95), decimals=0),
)


class GrowLog:
    """The readings of one year, at a fixed sampling delay.

    ``sampling`` is in seconds, between 1 and 3600; it is adjusted so that an
    hour holds a whole number of readings (``log.sampling`` is the value
    used). ``year`` defaults to the current one.
    """

    def __init__(self, year: int | None = None, sampling: float = 60.0, channels=CHANNELS) -> None:
        self.year = int(year or datetime.now().year)
        self.channels = tuple(channels)
        self.per_hour = int(min(max(round(3600.0 / float(sampling)), 1), 3600))
        self.sampling = 3600.0 / self.per_hour
        self.days = 366 if calendar.isleap(self.year) else 365
        self.start = datetime(self.year, 1, 1)
        self.size = self.days * 24 * self.per_hour  # readings in the year
        nothing = np.nan  # "no reading": never drawn, never counted
        self.raw = {c.key: np.full(self.size, nothing, dtype=np.float32) for c in self.channels}
        # lowest, average, highest — per hour and per day. What the far zooms read.
        self.hourly = {c.key: np.full((self.days * 24, 3), nothing, dtype=np.float32) for c in self.channels}
        self.daily = {c.key: np.full((self.days, 3), nothing, dtype=np.float32) for c in self.channels}
        self.last = -1  # the latest slot written
        self.version = 0  # grows with every change

    # ------------------------------------------------------------ writing
    def slot(self, when: datetime | None = None) -> int:
        """The number of the reading that the moment ``when`` belongs to."""
        when = when or datetime.now()
        return int((when - self.start).total_seconds() // self.sampling)

    def due(self, when: datetime | None = None) -> bool:
        """True when no reading was logged yet for the delay ``when`` is in."""
        slot = self.slot(when)
        return 0 <= slot < self.size and slot > self.last

    def add(self, when: datetime | None = None, **values: float) -> bool:
        """Log one reading of some or all channels: ``add(water_temp=20.4, light=1)``.

        Returns False (and logs nothing) when ``when`` is outside the year.
        """
        slot = self.slot(when)
        if not 0 <= slot < self.size:
            return False
        for key, value in values.items():
            self.raw[key][slot] = float(value)
        self.last = max(self.last, slot)
        hour = slot // self.per_hour
        self._summarize(hour, hour + 1)
        return True

    def fill(self, first: int = 0, **arrays) -> None:
        """Log many readings at once: arrays of values, starting at slot ``first``.

        For a file read back, or a simulation. Thousands of times faster than
        calling ``add`` in a loop.
        """
        count = 0
        for key, values in arrays.items():
            values = np.asarray(values, dtype=np.float32)[: self.size - first]
            self.raw[key][first : first + len(values)] = values
            count = max(count, len(values))
        if count:
            self.last = max(self.last, first + count - 1)
            self._summarize(first // self.per_hour, -(-(first + count) // self.per_hour))

    def _summarize(self, hour0: int, hour1: int) -> None:
        """Recompute the hourly and daily lowest / average / highest of some hours."""
        day0, day1 = hour0 // 24, -(-hour1 // 24)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # an hour with no reading is simply NaN
            for key, raw in self.raw.items():
                block = raw[hour0 * self.per_hour : hour1 * self.per_hour].reshape(-1, self.per_hour)
                hourly = self.hourly[key]
                hourly[hour0:hour1, 0] = np.nanmin(block, axis=1)
                hourly[hour0:hour1, 1] = np.nanmean(block, axis=1)
                hourly[hour0:hour1, 2] = np.nanmax(block, axis=1)
                block = hourly[day0 * 24 : day1 * 24].reshape(-1, 24, 3)
                daily = self.daily[key]
                daily[day0:day1, 0] = np.nanmin(block[:, :, 0], axis=1)
                daily[day0:day1, 1] = np.nanmean(block[:, :, 1], axis=1)
                daily[day0:day1, 2] = np.nanmax(block[:, :, 2], axis=1)
        self.version += 1

    # ------------------------------------------------------------ reading
    def day_index(self, month: int, day: int) -> int:
        """The number of a day in the year, from 0."""
        return (datetime(self.year, month, day) - self.start).days

    def day_readings(self, key: str, month: int, day: int) -> np.ndarray:
        """Every reading of one channel on one day (a view, not a copy)."""
        first = self.day_index(month, day) * 24 * self.per_hour
        return self.raw[key][first : first + 24 * self.per_hour]

    def alerts(self, month: int | None = None) -> int:
        """How many hours had a reading outside its safe range, in a month or the year."""
        if month is None:
            hours = slice(0, self.days * 24)
        else:
            first = self.day_index(month, 1) * 24
            hours = slice(first, first + calendar.monthrange(self.year, month)[1] * 24)
        bad = np.zeros(hours.stop - hours.start, dtype=bool)
        for channel in self.channels:
            if not channel.cycle:
                block = self.hourly[channel.key][hours]
                bad |= (block[:, 0] < channel.safe[0]) | (block[:, 2] > channel.safe[1])
        return int(bad.sum())

    # ------------------------------------------------------------ files
    def save(self, path) -> None:
        """Write the year to a compressed ``.npz`` file."""
        np.savez_compressed(path, _year=self.year, _sampling=self.sampling, _last=self.last, **self.raw)

    @classmethod
    def load(cls, path, channels=CHANNELS) -> GrowLog:
        """Read a year written by ``save``."""
        with np.load(path) as data:
            log = cls(int(data["_year"]), float(data["_sampling"]), channels)
            last = int(data["_last"])
            if last >= 0:
                log.fill(0, **{c.key: data[c.key][: last + 1] for c in log.channels if c.key in data})
        return log


# ----------------------------------------------------------------- colours
DARK = SimpleNamespace(
    panel=(0.06, 0.08, 0.09, 1.0),
    lane=(1.0, 1.0, 1.0, 0.04),  # the background of a lane
    ink=(0.86, 0.90, 0.92),
    muted=(0.50, 0.57, 0.60),
    accent=(0.05, 0.52, 0.57, 1.0),
    button=(0.02, 0.17, 0.19, 1.0),
    on_accent=(0.0, 0.0, 0.0),
    ok=(0.10, 0.62, 0.45),  # a day inside the safe range
    warn=(0.95, 0.70, 0.20),  # a day that touched a limit
    cold=(0.25, 0.50, 0.95),  # an average below the range
    hot=(0.92, 0.28, 0.30),  # an average above it; and every alert
    empty=(1.0, 1.0, 1.0, 0.05),  # a day without readings
    cursor=(1.0, 1.0, 1.0, 0.55),
)
LIGHT = SimpleNamespace(
    panel=(1.0, 1.0, 1.0, 0.92),
    lane=(0.0, 0.0, 0.0, 0.04),
    ink=(0.10, 0.10, 0.11),
    muted=(0.40, 0.42, 0.45),
    accent=(0.00, 0.37, 0.72, 1.0),
    button=(0.93, 0.94, 0.96, 1.0),
    on_accent=(1.0, 1.0, 1.0),
    ok=(0.20, 0.66, 0.42),
    warn=(0.93, 0.66, 0.12),
    cold=(0.22, 0.48, 0.92),
    hot=(0.84, 0.20, 0.18),
    empty=(0.0, 0.0, 0.0, 0.06),
    cursor=(0.0, 0.0, 0.0, 0.55),
)

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
PAD = 14.0  # inside the panel
LABELS = 96.0  # the column of lane names, on the left
VALUES = 168.0  # the column of numbers, on the right
ZOOM_STEP = math.log(1.3)  # how much wheel or pinch makes one zoom level


def _rects(x, y, w, h, colors) -> np.ndarray:
    """Many rectangles as instance rows: top-left corners, sizes, one colour each or one for all."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float32))
    rows = np.zeros((len(x), screen2d.SHAPE_FLOATS), dtype=np.float32)
    rows[:, 0] = x + 0.5 * np.asarray(w, dtype=np.float32)
    rows[:, 1] = np.asarray(y, dtype=np.float32) + 0.5 * np.asarray(h, dtype=np.float32)
    rows[:, 2] = 0.5 * np.asarray(w, dtype=np.float32)
    rows[:, 3] = 0.5 * np.asarray(h, dtype=np.float32)
    rows[:, 4:8] = colors
    return rows


def _number(value: float, channel: Channel) -> str:
    return "--" if not np.isfinite(value) else f"{value:.{channel.decimals}f}"


class GrowJournal:
    """The picture of a ``GrowLog``: year, month or day. One ``draw`` per frame.

    What you can read or set:

    * ``view`` — ``"year"``, ``"month"`` or ``"day"``;
    * ``month``, ``day`` — the date shown (1-12, 1-31);
    * ``focus`` — the key of the channel that colours the year view;
    * ``colors`` — ``hydro2d.DARK`` (default) or ``hydro2d.LIGHT``.
    """

    def __init__(self, log: GrowLog, *, colors=DARK, when: datetime | None = None) -> None:
        self.log = log
        self.colors = colors
        now = when or datetime.now()
        if now.year != log.year:  # another year: open on its last reading, or January
            now = log.start + timedelta(seconds=max(log.last, 0) * log.sampling)
        self.view = "month"
        self.month, self.day = now.month, now.day
        self.focus = next(c.key for c in log.channels if not c.cycle)
        self._zoom = 0.0  # wheel / pinch not yet turned into a zoom level

    # ------------------------------------------------------------ moving
    def step(self, direction: int) -> None:
        """Go to the previous (-1) or next (+1) month or day, whichever is shown."""
        if self.view == "month":
            self.month = min(max(self.month + direction, 1), 12)
            self.day = min(self.day, calendar.monthrange(self.log.year, self.month)[1])
        elif self.view == "day":
            index = min(max(self.log.day_index(self.month, self.day) + direction, 0), self.log.days - 1)
            date = self.log.start + timedelta(days=index)
            self.month, self.day = date.month, date.day

    def zoom_out(self) -> None:
        """Day -> month -> year."""
        self.view = {"day": "month", "month": "year"}.get(self.view, "year")

    def _channel(self, key: str) -> Channel:
        return next(c for c in self.log.channels if c.key == key)

    # ------------------------------------------------------------ drawing
    def draw(self, app, anchor, size) -> None:
        """Draw the journal in the rectangle ``anchor``, ``size`` and handle the mouse."""
        ui, c, log = app.aa, self.colors, self.log
        x0, y0 = float(anchor[0]), float(anchor[1])
        width, height = float(size[0]), float(size[1])
        ui.draw_screen_rect((x0, y0), (width, height), c.panel, 10)

        def button(x: float, y: float, w: float, name: str, on: bool = False, h: float = 26.0) -> bool:
            if on:
                return ui.draw_button((x, y), (w, h), name, c.accent, c.accent, c.on_accent, hover_text_color=c.on_accent)
            return ui.draw_button((x, y), (w, h), name, c.button, c.accent, c.ink, hover_text_color=c.on_accent)

        # --- the top line: where we are, and the way back up -------------
        x, y = x0 + PAD, y0 + PAD
        if button(x, y, 62, str(log.year), self.view == "year"):
            self.view = "year"
        if button(x + 66, y, 62, MONTHS[self.month - 1], self.view == "month"):
            self.view = "month"
        if button(x + 132, y, 44, f"{self.day:02d}", self.view == "day"):
            self.view = "day"
        if self.view != "year":
            if button(x + 188, y, 28, "<"):
                self.step(-1)
            if button(x + 220, y, 28, ">"):
                self.step(+1)
        alerts = log.alerts(None if self.view == "year" else self.month)
        note = f"{alerts} h in alert" if alerts else "no alert"
        ui.draw_screen_text(x + 262, y + 18, note, 11, c.hot if alerts else c.ok)

        body = (x0 + PAD, y + 26 + 12, width - 2 * PAD, height - 2 * PAD - 38)
        over = ui.pointer.over((body[0], body[1]), (body[2], body[3]))
        # Wheel or pinch over the journal: in, toward what is under the cursor; or out.
        self._zoom += math.log(max(app.zoom(), 1e-6)) if over else -self._zoom
        target = None  # (month, day or None) under the cursor, set by the view
        if self.view == "year":
            target, info = self._draw_year(app, button, body)
        elif self.view == "month":
            target, info = self._draw_lanes(app, body, month=True)
        else:
            target, info = self._draw_lanes(app, body, month=False)

        if info:  # what is under the cursor, written at the top right
            ui.draw_screen_text(x0 + width - PAD - ui.measure_screen_text(info, 11), y + 18, info, 11, c.ink)

        zoom_in = self._zoom > ZOOM_STEP or (over and app.clicked((body[0], body[1]), (body[2], body[3])))
        if zoom_in and target is not None and self.view != "day":
            month, day = target
            self.month = month
            self.view = "month" if day is None else "day"
            self.day = day or min(self.day, calendar.monthrange(log.year, month)[1])
            ui.pointer.redraw()
        elif self._zoom < -ZOOM_STEP and self.view != "year":
            self.zoom_out()
            ui.pointer.redraw()
        if abs(self._zoom) > ZOOM_STEP:
            self._zoom = 0.0

    # ---------------------------------------------------------------- year
    def _draw_year(self, app, button, body):
        ui, c, log = app.aa, self.colors, self.log
        bx, by, bw, bh = body
        # The channel that colours the squares: one small button each.
        x = bx
        for channel in log.channels:
            if not channel.cycle:
                w = ui.measure_screen_text(channel.name, 12) + 22
                if button(x, by, w, channel.name, channel.key == self.focus, h=22):
                    self.focus = channel.key
                x += w + 4
        channel = self._channel(self.focus)
        low, high = channel.safe

        top, names, legend = by + 22 + 24, 44.0, 26.0
        cell_w, cell_h = (bw - names) / 31.0, (bh - (top - by) - legend) / 12.0
        gap = 2.0
        for day in (1, 5, 10, 15, 20, 25, 30):  # the days, above the columns
            ui.draw_screen_text(bx + names + (day - 0.5) * cell_w - 4, top - 7, str(day), 10, c.muted)

        # One square per day of the year: 365 rectangles, one draw call.
        daily = log.daily[channel.key]
        dates = [log.start + timedelta(days=index) for index in range(log.days)]
        months = np.array([d.month for d in dates])
        days = np.array([d.day for d in dates])
        colors = np.empty((log.days, 4), dtype=np.float32)
        colors[:] = screen2d.rgba(c.empty)
        seen = np.isfinite(daily[:, 1])
        with np.errstate(invalid="ignore"):
            touched = seen & ((daily[:, 0] < low) | (daily[:, 2] > high))
            colors[seen] = (*c.ok, 1.0)
            colors[touched] = (*c.warn, 1.0)
            colors[seen & (daily[:, 1] < low)] = (*c.cold, 1.0)
            colors[seen & (daily[:, 1] > high)] = (*c.hot, 1.0)
        ui._chart_rects(
            _rects(bx + names + (days - 1) * cell_w, top + (months - 1) * cell_h, cell_w - gap, cell_h - gap, colors),
            sharp=True,
        )

        # The cursor: which month (its name, or its line), which day.
        cursor = ui.pointer.cursor
        row = int((cursor[1] - top) // cell_h) if cell_h > 0 else -1
        column = int((cursor[0] - bx - names) // cell_w) if cursor[0] >= bx + names else -1
        inside = 0 <= row < 12 and bx <= cursor[0] <= bx + bw
        for month in range(12):
            lit = inside and row == month
            ui.draw_screen_text(bx, top + (month + 0.5) * cell_h + 4, MONTHS[month], 11, c.ink if lit else c.muted)
        target, info = None, ""
        if inside:
            month = row + 1
            target = (month, None)
            if 0 <= column < calendar.monthrange(log.year, month)[1]:
                target = (month, column + 1)
                index = log.day_index(month, column + 1)
                ui.draw_screen_rect_outline(
                    (bx + names + column * cell_w - 1, top + row * cell_h - 1), (cell_w, cell_h), c.cursor, 2, 1.5
                )
                lo, mean, hi = daily[index]
                info = f"{column + 1} {MONTHS[row]}   {channel.name}  {_number(mean, channel)} {channel.unit}"
                if np.isfinite(mean):
                    info += f"   ({_number(lo, channel)} to {_number(hi, channel)})"

        # What the colours mean.
        x, y = bx + names, by + bh - 10
        for color, text in (
            (c.ok, f"in range {low:g} to {high:g} {channel.unit}"), (c.warn, "touched a limit"),
            (c.cold, "average too low"), (c.hot, "average too high"),
        ):  # fmt: skip
            ui.draw_screen_rect((x, y - 9), (10, 10), color, 2)
            x += 16 + ui.draw_screen_text(x + 16, y, text, 10, c.muted) + 18
        return target, info

    # ------------------------------------------------------- month and day
    def _draw_lanes(self, app, body, month: bool):
        """Every channel as a lane. ``month``: hour by hour; else one day at full sampling."""
        ui, c, log = app.aa, self.colors, self.log
        bx, by, bw, bh = body
        left, plot_w = bx + LABELS, bw - LABELS - VALUES
        axis = 18.0  # room for the day numbers or the hours, under the lanes
        if month:
            days = calendar.monthrange(log.year, self.month)[1]
            first = log.day_index(self.month, 1) * 24
            series = {ch.key: log.hourly[ch.key][first : first + days * 24].T for ch in log.channels}  # lo, mean, hi
            count = days * 24
        else:
            days = 1
            # At most two values per pixel: more cannot be seen, and costs.
            count = 24 * log.per_hour
            group = max(count // max(int(2 * plot_w), 1), 1)
            while count % group:
                group += 1
            count //= group
            series = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for ch in log.channels:
                    block = log.day_readings(ch.key, self.month, self.day).reshape(count, group)
                    series[ch.key] = np.stack((np.nanmin(block, 1), np.nanmean(block, 1), np.nanmax(block, 1)))
        step = plot_w / count  # the width of one column
        columns = left + np.arange(count) * step

        # --- the cursor: a column, and in a month the day it belongs to
        cursor = ui.pointer.cursor
        hover = int((cursor[0] - left) // step) if left <= cursor[0] < left + plot_w and by <= cursor[1] <= by + bh else None
        target, info = None, ""
        if hover is not None and month:
            day = hover // 24 + 1
            target = (self.month, day)
            info = f"{day} {MONTHS[self.month - 1]}  {hover % 24:02d}:00   click: open the day"
            ui.draw_screen_rect((left + (day - 1) * 24 * step, by), (24 * step, bh - axis), c.lane)
        elif hover is not None:
            seconds = int((hover + 0.5) * 86400 / count)
            info = f"{self.day} {MONTHS[self.month - 1]}  {seconds // 3600:02d}:{seconds % 3600 // 60:02d}"

        # --- the lanes: the two cycles are thin, the readings share the rest
        cycles = [ch for ch in log.channels if ch.cycle]
        levels = [ch for ch in log.channels if not ch.cycle]
        gap, thin = 6.0, 16.0
        tall = (bh - axis - len(cycles) * (thin + gap) - len(levels) * gap) / max(len(levels), 1)
        y = by
        hot = np.array((*c.hot, 0.9), dtype=np.float32)
        for ch in (*cycles, *levels):
            h = thin if ch.cycle else tall
            lo, mean, hi = series[ch.key]
            seen = np.isfinite(mean)
            ui.draw_screen_text(bx, y + 0.5 * h + 4, ch.name, 11, c.ink)
            ui.draw_screen_rect((left, y), (plot_w, h), c.lane)
            value_x = left + plot_w + 12
            if ch.cycle:
                # A stripe per column, as opaque as the share of the time it was on.
                on = seen & (mean > 0.02)
                colors = np.empty((int(on.sum()), 4), dtype=np.float32)
                colors[:, 0:3] = ch.color
                colors[:, 3] = mean[on]
                ui._chart_rects(_rects(columns[on], y, step + 0.5, h, colors), sharp=True)
                if hover is not None and seen[hover]:
                    text = ("on" if mean[hover] > 0.5 else "off") if not month else f"on {mean[hover]:.0%} of the hour"
                elif seen.any():
                    hours = float(mean[seen].sum()) * 24.0 * days / count
                    text = f"{hours / days:.1f} h on a day" if month else f"{hours:.1f} h on"
                    if not month:  # how many times it started
                        raw = log.day_readings(ch.key, self.month, self.day)
                        starts = int(np.sum((raw[1:] > 0.5) & ~(raw[:-1] > 0.5)))
                        text += f", {starts} starts"
                else:
                    text = "no reading"
                ui.draw_screen_text(value_x, y + 0.5 * h + 4, text, 11, c.muted)
            else:
                s0, s1 = ch.scale
                to_y = lambda v: y + h - np.clip((v - s0) / (s1 - s0), 0.0, 1.0) * h  # noqa: E731
                # The safe range, as a faint band behind the curve.
                safe_top, safe_bottom = float(to_y(ch.safe[1])), float(to_y(ch.safe[0]))
                ui.draw_screen_rect((left, safe_top), (plot_w, safe_bottom - safe_top), (*ch.color, 0.10))
                if seen.any():
                    with np.errstate(invalid="ignore"):
                        bad = seen & ((lo < ch.safe[0]) | (hi > ch.safe[1]))
                    xs = columns[seen] + 0.5 * step
                    colors = np.empty((len(xs), 4), dtype=np.float32)
                    colors[:] = (*ch.color, 0.30)
                    colors[bad[seen]] = hot
                    # Lowest to highest of each column: one thin line each, one draw call.
                    ui._chart_lines(
                        np.stack((xs, to_y(hi[seen])), 1), np.stack((xs, to_y(lo[seen])), 1), colors, max(min(step, 2.0), 1.0)
                    )
                    ui.draw_screen_polyline(np.stack((xs, to_y(mean[seen])), 1), ch.color, 1.5)
                if hover is not None and seen[hover]:
                    text = f"{_number(mean[hover], ch)} {ch.unit}"
                    if month:
                        text += f"  ({_number(lo[hover], ch)} to {_number(hi[hover], ch)})"
                    alert = lo[hover] < ch.safe[0] or hi[hover] > ch.safe[1]
                    ui.draw_screen_circle((columns[hover] + 0.5 * step, float(to_y(mean[hover]))), 3, ch.color)
                elif seen.any():
                    text = f"{_number(np.nanmean(mean), ch)} {ch.unit}  ({_number(np.nanmin(lo), ch)} to {_number(np.nanmax(hi), ch)})"
                    alert = bool(bad.any())
                else:
                    text, alert = "no reading", False
                ui.draw_screen_text(value_x, y + 0.5 * h + 4, text, 11, c.hot if alert else c.muted)
            y += h + gap

        # --- the axis under the lanes: days of the month, or hours of the day
        lanes_bottom = by + bh - axis
        if month:
            every = 1 if 24 * step >= 18 else 2 if 24 * step >= 9 else 5
            for day in range(1, days + 1, every):
                ui.draw_screen_text(left + (day - 0.5) * 24 * step - 4, lanes_bottom + 12, str(day), 10, c.muted)
            marks = left + np.arange(1, days) * 24 * step
        else:
            for hour in range(0, 25, 3):
                ui.draw_screen_text(left + hour / 24 * plot_w - 8, lanes_bottom + 12, f"{hour:02d}h", 10, c.muted)
            marks = left + np.arange(3, 24, 3) / 24 * plot_w
        # Faint lines between days (or every three hours), through all the lanes.
        ui._chart_rects(_rects(marks, by, 1.0, lanes_bottom - by - gap, np.array(screen2d.rgba(c.lane)) * (1, 1, 1, 2)), sharp=True)
        if hover is not None:
            ui.draw_screen_rect((columns[hover] + 0.5 * step - 0.5, by), (1.0, lanes_bottom - by - gap), c.cursor)
        return target, info
