"""Example 11 — a live dashboard: every data widget, fed by simulated sensors.

On the left, twelve widgets on a frosted panel: a pressure gauge, a wind
compass and the history of the wind speed, a water level, the torque of six
motors as bars and as a radar, an energy mix as a pie and as one stacked bar,
a signal meter, a grid of temperatures, a cloud of points.

On the right, the same kind of data in 3D: a field of bars, a thick pie and a
glass tank, lit and casting shadows. Drag there to turn around them.

From the project root (after running the installer in toolchain/):

    .venv/bin/python examples/11_dashboard/main.py
    .venv/bin/python examples/11_dashboard/main.py --raymarch     # soft shadows in the 3D part

Keys: Space pauses the sensors.

How to read the code: every widget is ONE call with the same three things —
where (``anchor``, ``size``), the data, the colours. The values come from
``read_sensors``; replace that function with your own readings and the
dashboard is yours.

The picture changes all the time, so the last line of ``draw_frame`` asks for
the next frame (``app.request_redraw()``): without it the program would wait
for an event, as every program of this project does.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

# The drawing library is the graphics/ folder of the project.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "graphics"))

import app3d  # noqa: E402
from charts2d import History  # noqa: E402

BLUE = (0.20, 0.60, 1.00, 1.0)
CYAN = (0.20, 0.85, 0.85, 1.0)
ORANGE = (1.00, 0.55, 0.10, 1.0)
GREEN = (0.30, 0.80, 0.40, 1.0)
RED = (0.90, 0.25, 0.35, 1.0)
PURPLE = (0.65, 0.40, 0.95, 1.0)
YELLOW = (1.00, 0.85, 0.20, 1.0)
WATER = (0.15, 0.55, 0.95, 1.0)
FLOOR = (0.30, 0.31, 0.36)
PANEL = (0.07, 0.07, 0.10, 0.55)  # the tint of the frosted panel
TITLE = (0.62, 0.66, 0.76)
VALUE = (0.92, 0.94, 0.98)

ENERGY = [GREEN, YELLOW, BLUE, ORANGE]  # one colour per source: wind, sun, water, gas
# A scale for the water: red when nearly empty, blue when comfortable.
LEVEL_STOPS, LEVEL_COLORS = [0.0, 0.2, 0.4, 1.0], [RED, ORANGE, WATER, WATER]


def read_sensors(t: float) -> dict:
    """Pretend readings at time ``t`` (seconds). Replace with real ones."""
    motors = np.arange(6)
    grid_y, grid_x = np.mgrid[0:6, 0:10]
    return {
        "pressure": 5.0 + 3.2 * math.sin(0.7 * t) + 0.6 * math.sin(3.1 * t),  # bar, 0..10
        "wind_speed": 9.0 + 6.0 * math.sin(0.5 * t) + 2.0 * math.sin(2.3 * t + 1.0),  # m/s, 0..20
        "wind_from": (140.0 + 60.0 * math.sin(0.23 * t)) % 360.0,  # degrees
        "water": 0.5 + 0.45 * math.sin(0.35 * t),  # 0..1
        "torques": 40.0 + 30.0 * np.sin(0.9 * t + motors * 1.1) + 8.0 * np.sin(3.0 * t + motors),  # N·m, 0..80
        "energy": np.array([30 + 20 * math.sin(0.3 * t), 25 + 20 * math.sin(0.2 * t + 2), 20.0, 15 + 10 * math.cos(0.4 * t)]),
        "signal": 0.5 + 0.5 * math.sin(4.0 * t) * math.sin(0.6 * t),  # 0..1
        "temperatures": 20.0 + 6.0 * np.sin(0.5 * grid_x + 0.8 * t) * np.cos(0.7 * grid_y - 0.5 * t),  # °C
    }


def main() -> None:
    app = app3d.App("Example 11 — dashboard", size=(1280, 780))
    ui = app.aa
    app.camera.distance = 13.0
    app.camera.target[:] = (-3.2, 0.0, 1.0)  # look to the left of the 3D objects: they appear on the right
    app.camera.yaw, app.camera.pitch = -1.05, 0.42

    wind_history = History(360)  # the last 360 readings of the wind speed
    pressure_history = History(360)
    cloud = np.random.default_rng(1).normal(size=(400, 2))  # points for the scatter
    started = time.perf_counter()
    clock, paused = 0.0, False
    last = started
    frames, shown_rate, rate_time = 0, 0.0, started

    def title(x: float, y: float, text: str) -> None:
        ui.draw_screen_text(x, y, text, 11, TITLE)

    def draw_frame() -> None:
        nonlocal clock, paused, last, frames, shown_rate, rate_time
        now = time.perf_counter()
        if app.key_pressed("SPACE"):
            paused = not paused
        if not paused:
            clock += now - last
        last = now
        frames += 1
        if now - rate_time >= 0.5:  # the frame rate, refreshed twice a second
            shown_rate, frames, rate_time = frames / (now - rate_time), 0, now

        data = read_sensors(clock)
        if not paused:
            wind_history.add(data["wind_speed"])
            pressure_history.add(data["pressure"])

        width, height = app.size()
        if app.cursor()[0] > 640:  # the mouse moves the camera only over the 3D part
            app.handle_camera()

        # ================= the 3D part: three widgets that are solids =================
        ui.set_light((0.5, -0.4, 0.9), softness=0.2)
        ui.draw_box((2.6, 0.0, -0.06), (9.0, 9.0, 0.1), FLOOR, specular=0.0)
        # A field of bars: the grid of temperatures, each column its own colour.
        heights = data["temperatures"][:4, :6] - 12.0
        ui.draw_bars_3d((0.0, 0.8, 0.0), (3.6, 2.4, 2.2), heights, [BLUE, CYAN, GREEN, YELLOW, ORANGE, RED], maximum=16.0)
        # A pie with thickness: the energy mix, slices pulled slightly apart.
        ui.draw_pie_3d((4.6, -1.6, 0.0), 1.3, 0.45, data["energy"], ENERGY, explode=0.08)
        # A glass tank: the water level. Last, as every see-through solid.
        ui.draw_tank_3d((1.2, -2.2, 0.0), 0.8, 2.4, data["water"], color=WATER)

        # ================= the 2D part, on a frosted panel =================
        ui.draw_screen_acrylic_rect((12, 12), (620, height - 24), PANEL, radius=18, blur=16)

        # --- row 1: one value on a round scale, a direction, a level, a meter
        title(30, 40, "PRESSURE")
        ui.draw_gauge((30, 48), (150, 150), data["pressure"], 0, 10, label=f"{data['pressure']:.1f} bar")

        title(205, 40, "WIND FROM")
        ui.draw_compass((205, 48), (150, 150), data["wind_from"], ORANGE)

        title(380, 40, "WATER")
        ui.draw_level(
            (380, 48), (64, 150), data["water"], stops=LEVEL_STOPS, colors=LEVEL_COLORS, label=f"{data['water'] * 100:.0f}%"
        )
        title(470, 40, "SIGNAL")
        ui.draw_meter((470, 48), (26, 150), data["signal"], segments=14)
        title(520, 40, "BATTERY")
        ui.draw_level((520, 48), (96, 26), 1.0 - 0.5 * data["water"], color=GREEN)  # wider than tall: it fills sideways
        title(520, 100, "ENERGY MIX")
        ui.draw_stacked_bar((520, 108), (96, 18), data["energy"], ENERGY)
        title(520, 152, f"{shown_rate:.0f} frames/s")

        # --- row 2: history. Two curves that scroll as readings arrive.
        title(30, 226, f"WIND SPEED   {wind_history.last:.1f} m/s")
        ui.draw_line_chart((30, 234), (285, 96), wind_history.values, CYAN, minimum=0, maximum=20, fill=0.25, grid=3)
        title(332, 226, f"PRESSURE   {pressure_history.last:.1f} bar")
        ui.draw_line_chart((332, 234), (285, 96), pressure_history.values, ORANGE, minimum=0, maximum=10, fill=0.25, grid=3)

        # --- row 3: several values of the same kind, compared
        title(30, 360, "MOTOR TORQUES (N.m)")
        ui.draw_bars((30, 368), (190, 130), data["torques"], [BLUE, CYAN, GREEN, YELLOW, ORANGE, RED], maximum=80)
        title(240, 360, "THE SAME, AS A RADAR")
        ui.draw_radar((240, 368), (150, 130), data["torques"], PURPLE, maximum=80)
        title(410, 360, "ENERGY MIX")
        ui.draw_pie((410, 368), (130, 130), data["energy"], ENERGY, hole=0.55, gap=2)
        for index, name in enumerate(("wind", "sun", "water", "gas")):
            ui.draw_screen_circle((556, 386 + 22 * index), 5, ENERGY[index])
            ui.draw_screen_text(568, 390 + 22 * index, name, 11, VALUE)

        # --- row 4: many values at once
        title(30, 530, "TEMPERATURES (60 sensors)")
        ui.draw_heatmap((30, 538), (285, 150), data["temperatures"], minimum=14, maximum=26, gap=2)
        title(332, 530, "400 POINTS")
        spin = 0.3 * clock
        turned = cloud @ np.array(((math.cos(spin), -math.sin(spin)), (math.sin(spin), math.cos(spin))))
        ui.draw_scatter((332, 538), (285, 150), turned[:, 0] * 1.6, turned[:, 1], PURPLE, x_range=(-4, 4), y_range=(-3, 3), point=2.5)

        ui.draw_screen_text(30, height - 28, "Space: pause the sensors      drag on the right: turn the 3D scene", 11, TITLE)
        app.request_redraw()  # live data: ask for the next frame

    app.run(draw_frame)


if __name__ == "__main__":
    main()
