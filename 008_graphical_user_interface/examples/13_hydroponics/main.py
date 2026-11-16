"""Example 13 — the journal of an indoor hydroponic culture, over a year.

One widget fills the window: the readings of a hydroponic system since the
first of January — light cycle, watering cycle, temperature of the water and
of the air, humidity, pH, strength of the nutrient solution (EC), level of
the tank — at three zooms: the year, a month, a day.

From the project root (after running the installer in toolchain/):

    .venv/bin/python examples/13_hydroponics/main.py
    .venv/bin/python examples/13_hydroponics/main.py --speed 600      # ten minutes of culture per second
    .venv/bin/python examples/13_hydroponics/main.py --sampling 10    # a reading every ten seconds
    .venv/bin/python examples/13_hydroponics/main.py --light          # light colours
    .venv/bin/python examples/13_hydroponics/main.py --save year.npz  # keep the year when closing
    .venv/bin/python examples/13_hydroponics/main.py --load year.npz  # and open it again

Mouse: click a month or a day to zoom in; the buttons at the top (year,
month, day) zoom back out; ``<`` ``>`` go to the neighbour. Wheel or pinch
zooms in and out at the cursor.
Keys: Left / Right the neighbour, Backspace zooms out.

The whole program is four lines of ``draw_frame``:

    if log.due(now):                      # has the sampling delay passed?
        log.add(now, **read_sensors(now)) # then one reading goes in the log
    journal.draw(app, anchor, size)       # and the journal shows the year

There are no sensors here: ``simulate`` invents a plausible culture, with a
few accidents (a chiller that fails in July, a tank left to run low, a pH that
drifts) so the alerts have something to show. Replace ``read_sensors`` with
your own readings and the journal is yours.

**Sleeping between readings.** Nothing is drawn while nothing happens. A small
background thread wakes the program once per sampling delay — once a minute
by default — for the one frame in which the new reading is logged and shown.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# The drawing library is the graphics/ folder of the project.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "graphics"))

import app3d  # noqa: E402
import hydro2d  # noqa: E402
from hydro2d import GrowJournal, GrowLog  # noqa: E402

MUTED = (0.50, 0.57, 0.60)


def simulate(log: GrowLog, slots: np.ndarray) -> dict[str, np.ndarray]:
    """Invented readings for the given slots of the year (0 = 1 January, 00:00).

    One formula per channel, written for whole arrays so that a year is
    computed in a blink; the same function gives the single live reading.
    """
    slots = np.asarray(slots, dtype=np.float64)
    day = slots / (24 * log.per_hour)  # days since 1 January, with decimals
    hour = (day % 1.0) * 24.0
    whole = np.floor(day)
    noise = np.random.default_rng(int(slots[0]) + len(slots))

    # Lights: 18 hours a day while the plants grow, 12 from September (flowering).
    growing = whole < 243
    light = np.where(growing, hour >= 6.0, (hour >= 8.0) & (hour < 20.0))
    # Watering: 15 minutes every 2 hours under the lights, 10 minutes every 4 hours in the dark.
    pump = np.where(light, (hour % 2.0) < 0.25, (hour % 4.0) < 10.0 / 60.0)

    season = np.sin(np.pi * day / 365.0) ** 2  # 0 in winter, 1 in midsummer
    warm = 0.5 - 0.5 * np.cos(2.0 * np.pi * (hour - 5.0) / 24.0)  # the day's warmth, peaking mid-afternoon
    water = 19.2 + 1.3 * warm + 1.2 * season + noise.normal(0.0, 0.08, len(slots))
    water += np.where((whole >= 196) & (whole < 200), 4.0, 0.0)  # July: the chiller fails for four days
    water -= np.where((whole >= 20) & (whole < 22), 2.6, 0.0)  # January: a cold snap
    air = 22.0 + 3.0 * light + 1.3 * warm + 0.8 * season + noise.normal(0.0, 0.15, len(slots))
    humidity = 66.0 - 7.0 * light - 3.0 * warm + noise.normal(0.0, 0.8, len(slots))
    humidity += np.where((whole >= 130) & (whole < 134), 16.0, 0.0)  # May: the dehumidifier is unplugged

    # pH creeps up as the plants feed and is brought back down every week or so.
    ph = 5.7 + (day * 0.09) % 0.65 + noise.normal(0.0, 0.01, len(slots))
    ph += np.where((whole >= 280) & (whole < 285), (day - 280.0) * 0.12, 0.0)  # October: nobody corrects it
    # EC falls as nutrients are taken up; the tank is refreshed every week.
    ec = 2.1 - 0.7 * ((day % 7.0) / 7.0) + noise.normal(0.0, 0.01, len(slots))
    # The tank empties in five days and is topped up; once, in late October, it is forgotten.
    level = 95.0 - 52.0 * ((day % 5.0) / 5.0)
    level = np.where((whole >= 300) & (whole < 304), 43.0 - (day - 300.0) * 6.5, level)
    return {
        "light": light.astype(np.float32), "pump": pump.astype(np.float32), "water_temp": water, "air_temp": air,
        "humidity": humidity, "ph": ph, "ec": ec, "level": level,
    }  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sampling", type=float, default=60.0, help="seconds between two readings (default 60)")
    parser.add_argument("--speed", type=float, default=1.0, help="how many times faster than a clock the culture runs")
    parser.add_argument("--light", action="store_true", help="light colours")
    parser.add_argument("--save", help="write the year to this .npz file when closing")
    parser.add_argument("--load", help="open a year written by --save instead of simulating one")
    app = app3d.App("Example 13 — hydroponics", size=(1100, 680), parser=parser)
    ui = app.aa
    args = app.args

    opened = datetime.now()
    if args.load:
        log = GrowLog.load(args.load)
    else:
        log = GrowLog(sampling=args.sampling)
        # The year so far, in one call: every reading from 1 January to this minute.
        log.fill(0, **simulate(log, np.arange(log.slot(opened) + 1)))
    journal = GrowJournal(log, colors=hydro2d.LIGHT if args.light else hydro2d.DARK)
    if args.light:
        app.gpu.clear_color = (0.90, 0.92, 0.94, 1.0)

    started = time.perf_counter()

    def culture_time() -> datetime:
        """The clock of the culture: the real one, or faster with --speed."""
        return opened + timedelta(seconds=(time.perf_counter() - started) * args.speed)

    def read_sensors(when: datetime) -> dict[str, float]:
        """One reading of every channel. Replace with your own sensors."""
        return {key: float(values[0]) for key, values in simulate(log, np.array([log.slot(when)])).items()}

    # Wake the sleeping program once per sampling delay, for the frame that logs a reading.
    def wake() -> None:
        while True:
            time.sleep(max(log.sampling / args.speed, 0.05))
            app.request_redraw()

    threading.Thread(target=wake, daemon=True).start()

    def draw_frame() -> None:
        now = culture_time()
        if log.due(now):  # has the sampling delay passed?
            log.add(now, **read_sensors(now))

        for key in app.keys_pressed(repeat=True):
            if key in ("LEFT", "RIGHT"):
                journal.step(1 if key == "RIGHT" else -1)
            elif key == "BACKSPACE":
                journal.zoom_out()

        width, height = app.size()
        journal.draw(app, (20, 20), (width - 40, height - 64))
        ui.draw_screen_text(
            24, height - 18,
            f"{now:%d %b %Y  %H:%M}    one reading every {log.sampling:g} s    "
            "click: zoom in    Backspace: zoom out    Left / Right: neighbour",
            11, journal.colors.muted,
        )  # fmt: skip

    app.run(draw_frame)
    if args.save:
        log.save(args.save)
        print(f"year written to {args.save}")


if __name__ == "__main__":
    main()
