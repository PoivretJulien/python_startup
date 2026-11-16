"""Example 12 — three pages, buttons to go from one to the other, and a table.

A small program that looks after forty devices, dressed as a desktop
application, light or dark: a soft picture behind frosted glass, cards of
glass, and one blue accent. Three buttons at the top open its three pages:

* Overview — how many devices are fine, off, or in alarm;
* Devices  — the list, as a table: striped, scrollable, a line can be chosen;
* Settings — how the table looks.

From the project root (after running the installer in toolchain/):

    .venv/bin/python examples/12_pages/main.py
    .venv/bin/python examples/12_pages/main.py --image photo.jpg   # your own background

Mouse: click the buttons; on the Devices page move over the table, click a
line, scroll with two fingers or the wheel.
Keys: Left / Right (or Tab) change page; Up / Down choose a line.

Two calls do the work, and both are ordinary draw calls:

    if ui.draw_button(anchor, size, "Devices"):     # True once per click
        page = "Devices"

    line = ui.draw_table(anchor, size, rows, text_color, [COLOR_A, COLOR_B])

**A page is a function.** The variable ``page`` holds a name; ``draw_frame``
draws the buttons, then calls the function of that name. Going to another
page is changing the variable — there is nothing to open, close or hide.

**The look is a handful of colours**, all at the top of this file. The
background is a picture behind a sheet of frosted glass: the picture is drawn
first, then ``draw_screen_acrylic_rect`` over the whole window blurs it and
lays a light tint on it. On it, a *card* is a second, smaller piece of frosted
glass — whiter, and blurring more — with a hairline around it, and the table
is made of the same glass (``blur=`` and see-through stripes); sliders on the
Settings page change how much it blurs, how white it is, and how much the
background itself is blurred. A button is
glass too (``blur=``), with the same hairline, and the one that is "on" wears
the accent.

**Two themes.** Every colour the pages use is read from one object, ``theme``
— ``theme.ink``, ``theme.card`` ... ``LIGHT`` and ``DARK`` are two such objects
with the same names; the "Dark theme" button of the Settings page swaps one
for the other, and the next frame is drawn in the other colours.

**One factor for all the gaps.** No position of this file is a bare number
where two things meet: the distance goes through ``gap(pixels)``, which
multiplies it by the Spacing slider of the Settings page. Sizes stay; only
what is *between* things moves, and whatever is flexible (the table, the
three small cards) takes the room that is left.

**Nothing runs while nothing happens.** The program draws a frame when the
mouse moves, a button is clicked or a key is pressed, and sleeps otherwise:
watch the "frames drawn" counter at the bottom stand still when you take your
hand off the mouse.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

# The drawing library is the graphics/ folder of the project.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "graphics"))

import app3d  # noqa: E402
import numpy as np  # noqa: E402
from texture3d import Texture  # noqa: E402
from widgets2d import Slider  # noqa: E402

# ---- two themes: the same names, other colours ------------------------------
# Everything the pages draw takes its colours from one of these two objects
# (`theme.ink`, `theme.card` ...). Switching theme is choosing the other one.
LIGHT = SimpleNamespace(
    dark=False,
    ink=(0.10, 0.10, 0.11),  # text
    muted=(0.38, 0.39, 0.42),  # small titles, hints
    green=(0.06, 0.49, 0.18, 1.0),
    red=(0.77, 0.17, 0.11, 1.0),
    grey=(0.56, 0.57, 0.60, 1.0),
    card=(1.0, 1.0, 1.0),  # the tint of a card; how much it covers is a slider
    border=(0.0, 0.0, 0.0, 0.10),  # the hairline around cards and buttons
    button=(0.99, 0.99, 0.99),  # a button at rest
    button_over=(0.94, 0.95, 0.96),  # the same button under the cursor
    rows=((1.0, 1.0, 1.0), (0.88, 0.90, 0.94)),  # the two stripes of the table
    header=(0.82, 0.85, 0.90),
    track=(0.0, 0.0, 0.0, 0.12),  # the empty part of a level, the line of a slider
    hover=(0.0, 0.0, 0.0, 0.05),  # laid over the line of the table under the cursor
    glass=(0.95, 0.96, 0.97, 0.45),  # the frosted glass over the picture: nearly white, thin
)
DARK = SimpleNamespace(
    dark=True,
    ink=(0.95, 0.95, 0.96),
    muted=(0.68, 0.70, 0.74),
    green=(0.40, 0.80, 0.47, 1.0),
    red=(1.00, 0.52, 0.47, 1.0),
    grey=(0.52, 0.54, 0.58, 1.0),
    card=(0.03, 0.035, 0.045),  # darker than the window's glass: a card sinks in, it does not glow
    border=(1.0, 1.0, 1.0, 0.10),
    button=(0.20, 0.21, 0.24),
    button_over=(0.29, 0.30, 0.34),
    rows=((0.03, 0.035, 0.045), (0.11, 0.12, 0.14)),
    header=(0.0, 0.0, 0.0),
    track=(1.0, 1.0, 1.0, 0.16),
    hover=(1.0, 1.0, 1.0, 0.07),
    glass=(0.10, 0.11, 0.13, 0.50),  # the same glass, smoked: a dark grey, not black
)
BLUE = (0.00, 0.37, 0.72, 1.0)  # the accent: the button that is "on", the chosen line
TEAL = (0.02, 0.48, 0.53, 1.0)  # two other accents, offered on the Settings page
PURPLE = (0.42, 0.31, 0.78, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)  # the name of a button that wears the accent, in both themes
BUTTON_W, BUTTON_H = 128, 30  # wide and low
BAR_PAD = 4  # between the box around the three page buttons and the buttons
MARGIN, RIGHT = 40, 860  # the left and right edges of everything

GLASS_BLUR = 10  # pixels, at the start (a slider): light, so that the stronger blur of the cards shows

PAGES = ("Overview", "Devices", "Settings")
STATES = ("online", "off", "alarm")
ROOMS = ("hall", "lab", "roof", "cellar", "workshop")
KINDS = ("pump", "fan", "valve", "heater", "sensor")


def make_devices() -> list[dict]:
    """Forty pretend devices. Replace with your own list."""
    devices = []
    for number in range(40):
        state = "alarm" if number % 11 == 5 else "off" if number % 7 == 3 else "online"
        load = 0 if state == "off" else int(50 + 45 * math.sin(number * 1.7))
        devices.append({"name": f"{KINDS[number % 5]} {number + 1:02d}", "room": ROOMS[number % 3 + number % 2], "state": state, "load": load})
    return devices


def make_picture(width: int = 450, height: int = 300) -> np.ndarray:
    """A picture made of a few large soft-coloured petals, as ``(height, width, 3)`` bytes.

    Stands in for a wallpaper, so the example needs no file.
    """
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    picture = np.empty((height, width, 3), dtype=np.float32)
    picture[:] = (0.80, 0.88, 0.97)  # a pale sky
    shapes = (
        (np.hypot(x - 120, y - 330) < 260, (0.30, 0.56, 0.95)),  # a large blue petal
        (np.hypot(x - 300, y - 360) < 230, (0.14, 0.38, 0.86)),  # a deeper one
        (np.hypot(x - 430, y - 300) < 170, (0.50, 0.42, 0.90)),  # a violet one
        (np.hypot(x - 380, y - 20) < 90, (1.00, 0.86, 0.72)),  # a warm glow
        (np.hypot(x - 20, y - 40) < 110, (0.62, 0.90, 0.92)),  # a mint one
    )
    for where, color in shapes:
        picture[where] = color
    return (picture * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", help="a picture file for the background")
    app = app3d.App("Example 12 — pages", size=(900, 600), parser=parser)
    ui = app.aa
    # Sent to the graphics card once; every frame only shows it.
    picture = Texture(app.args.image or make_picture(), app.gpu)
    devices = make_devices()

    page = "Overview"  # the page shown: one of PAGES
    selected = 0  # the device chosen in the table
    scroll = 0.0  # the first line shown by the table (a float: gestures are smooth)
    striped = True
    accent = BLUE
    theme = LIGHT  # or DARK: a button of the Settings page switches
    row_height = Slider((330, 196), (610, 196), 20, 40, 26, step=1, line_color=theme.track, handle_color=BLUE, hover_color=theme.ink)
    # The glass of the cards: how far it blurs what is behind (0 = a plain
    # panel), and how much of its white covers the result (1 = solid white).
    card_blur = Slider((330, 286), (610, 286), 0, 40, 26, step=1, line_color=theme.track, handle_color=BLUE, hover_color=theme.ink)
    card_white = Slider((330, 331), (610, 331), 0.15, 1.0, 0.5, line_color=theme.track, handle_color=BLUE, hover_color=theme.ink)
    # The glass over the whole window, far behind everything: how much it blurs the picture.
    deep_blur = Slider((330, 376), (610, 376), 0, 40, GLASS_BLUR, step=1, line_color=theme.track, handle_color=BLUE, hover_color=theme.ink)
    # One factor for every gap of the interface: between two buttons, two
    # cards, two lines of settings. 1 is the design; 0 glues everything.
    spacing = Slider((330, 0), (610, 0), 0.0, 2.0, 1.0, line_color=theme.track, handle_color=BLUE, hover_color=theme.ink)
    frames = 0

    def state_color(state: str):
        """Green, grey or red — in the shade of the current theme."""
        return {"online": theme.green, "off": theme.grey, "alarm": theme.red}[state]

    def gap(pixels: float) -> float:
        """A gap of the design, multiplied by the spacing factor.

        Every distance *between* two things goes through here; the sizes of
        the things themselves do not. That is the whole trick: one slider
        then tightens or loosens the entire interface.
        """
        return round(pixels * spacing.value)

    def top() -> float:
        """Where the page starts: under the row of buttons."""
        return MARGIN + BUTTON_H + 2 * BAR_PAD + gap(18)

    def button(anchor, size, name: str, on: bool = False, color=None, enabled: bool = True) -> bool:
        """One button of this program; ``True`` when clicked.

        At rest: frosted glass of the theme's colour, a hairline around it, its name in dark text.
        ``on=True`` (the page shown, a switch that is on): the accent, its
        name in white. ``color`` replaces the accent for that button.
        """
        color = color or accent
        # Glass like the cards: the same blur, a little more covering so the
        # button stands out on a card. The colour keeps its hue; only how much
        # it covers follows the slider.
        blur, cover = card_blur.value, min(card_white.value + 0.25, 1.0)
        if on:
            body = (*color[:3], max(cover, 0.8))  # the accent stays strong enough for white text
            return ui.draw_button(anchor, size, name, body, body, WHITE, hover_text_color=WHITE, blur=blur, enabled=enabled)
        return ui.draw_button(
            anchor, size, name, (*theme.button, cover), (*theme.button_over, min(cover + 0.15, 1.0)), theme.ink,
            hover_text_color=tuple(0.5 + 0.5 * c for c in color[:3]) if theme.dark else color[:3], border_color=theme.border, blur=blur, enabled=enabled,
        )  # fmt: skip

    def table_look() -> dict:
        """The colours of a table: glass like the cards, tinted by the stripes."""
        white = card_white.value
        return {
            "backgrounds": [(*theme.rows[0], white), (*theme.rows[1], white)] if striped else [(*theme.rows[0], white)],
            "header_color": theme.muted,
            "header_background": (*theme.header, min(white + 0.15, 1.0)),
            "blur": card_blur.value,
        }

    def card(anchor, size) -> None:
        """Frosted glass with round corners and a hairline: where content sits."""
        if card_blur.value > 0:
            ui.draw_screen_acrylic_rect(anchor, size, (*theme.card, card_white.value), radius=8, blur=card_blur.value)
        else:  # no blur asked: an ordinary see-through rectangle
            ui.draw_screen_rect(anchor, size, (*theme.card, card_white.value), 8)
        ui.draw_screen_rect_outline(anchor, size, theme.border, 8, 1)

    def go_to(name: str) -> None:
        nonlocal page
        page = name

    # ------------------------------------------------------------ page 1
    def overview() -> None:
        counts = [sum(device["state"] == state for device in devices) for state in STATES]
        y = top()
        small, lower = 110, 164  # the heights of the three small cards and of the card under them
        # On the left, one tall card: as tall as the two rows on its right.
        card((MARGIN, y), (270, small + gap(16) + lower))
        ui.draw_screen_text(MARGIN + 18, y + 28, "ALL DEVICES", 11, theme.muted)
        ui.draw_pie((MARGIN + 25, y + 46), (220, 220), counts, [state_color(state) for state in STATES], hole=0.62, gap=2)
        total = f"{len(devices)}"
        ui.draw_screen_text(MARGIN + 135 - 0.5 * ui.measure_screen_text(total, 30), y + 171, total, 30, theme.ink)  # centred in the ring

        # On its right, three cards share the width that the gaps leave.
        x0 = MARGIN + 270 + gap(20)
        width = (RIGHT - x0 - 2 * gap(10)) / 3
        for index, state in enumerate(STATES):
            x = x0 + (width + gap(10)) * index
            card((x, y), (width, small))
            ui.draw_screen_circle((x + 22, y + 26), 4, state_color(state))
            ui.draw_screen_text(x + 34, y + 30, state.upper(), 11, theme.muted)
            ui.draw_screen_text(x + 18, y + 86, f"{counts[index]}", 30, theme.ink)

        # A button inside a page can open another page too.
        y += small + gap(16)
        card((x0, y), (RIGHT - x0, lower))
        ui.draw_screen_text(x0 + 20, y + 30, "CHOSEN IN THE LIST", 11, theme.muted)
        device = devices[selected]
        ui.draw_screen_text(x0 + 20, y + 68, device["name"], 18, theme.ink)
        ui.draw_screen_text(x0 + 20, y + 96, f"{device['room']}   {device['state']}   load {device['load']} %", 12, theme.muted)
        if button((x0 + 20, y + 116), (190, BUTTON_H), "Show it in the list"):
            go_to("Devices")

    # ------------------------------------------------------------ page 2
    def device_list() -> None:
        nonlocal selected, scroll
        side = 240  # the width of the card on the right
        anchor = (MARGIN, top())
        size = (RIGHT - side - gap(20) - MARGIN, 540 - top())  # the table takes what the gap leaves
        height = row_height.value
        visible = int((size[1] - height) // height)  # lines that fit under the header
        last = max(len(devices) - visible, 0)

        # Scrolling: a two-finger slide or the wheel moves the first line shown.
        if ui.in_screen_rect(app.cursor(), anchor, size):
            scroll -= app.pan()[1] / height
            scroll -= 3.0 * math.log(app.zoom()) / math.log(1.1)
        # Up / Down choose a line; the table follows so the line stays in view.
        for key in app.keys_pressed(repeat=True):
            if key in ("UP", "DOWN"):
                selected = min(max(selected + (1 if key == "DOWN" else -1), 0), len(devices) - 1)
                scroll = min(max(scroll, selected - visible + 1), selected)
        scroll = min(max(scroll, 0.0), last)

        # The list as lines of text. A line in alarm is written in red:
        # text_color takes one colour, or one per line.
        rows = [(d["name"], d["room"], d["state"], f"{d['load']} %") for d in devices]
        inks = [theme.red if d["state"] == "alarm" else theme.grey if d["state"] == "off" else theme.ink for d in devices]
        line = ui.draw_table(
            anchor, size, rows, inks, **table_look(), header=("NAME", "ROOM", "STATE", "LOAD"),
            columns=(3, 2, 2, 1), align=("left", "left", "left", "right"),
            row_height=height, first=round(scroll), hover_color=theme.hover,
            selected=selected, selected_color=accent,
        )  # fmt: skip
        ui.draw_screen_rect_outline(anchor, (size[0], (visible + 1) * height), theme.border, 0, 1)
        if line is not None and app.clicked(anchor, size):  # a click on the line under the cursor
            selected = line

        # --- on the right: the chosen device, and two buttons that change it
        device = devices[selected]
        x, y = RIGHT - side, top()
        card((x, y), (side, 210))
        ui.draw_screen_text(x + 20, y + 34, device["name"], 18, theme.ink)
        ui.draw_screen_text(x + 20, y + 64, f"room: {device['room']}", 12, theme.muted)
        ui.draw_screen_circle((x + 24, y + 82), 4, state_color(device["state"]))
        ui.draw_screen_text(x + 36, y + 86, device["state"], 12, theme.ink)
        ui.draw_screen_text(x + 20, y + 112, f"load: {device['load']} %", 12, theme.muted)
        ui.draw_level((x + 20, y + 124), (side - 40, 8), device["load"], 0, 100, color=accent, track=theme.track, radius=4)
        half = (side - 40 - gap(4)) / 2  # two buttons share the width, less the gap between them
        if button((x + 20, y + 156), (half, BUTTON_H), "Start", color=theme.green, enabled=device["state"] != "online"):
            device.update(state="online", load=60)
        if button((x + 20 + half + gap(4), y + 156), (half, BUTTON_H), "Stop", color=theme.red, enabled=device["state"] != "off"):
            device.update(state="off", load=0)
        y += 210 + gap(20)
        ui.draw_screen_text(x + 2, y + 8, "click a line, or Up / Down", 11, theme.muted)
        ui.draw_screen_text(x + 2, y + 28, "two fingers or wheel: scroll", 11, theme.muted)

    # ------------------------------------------------------------ page 3
    def settings() -> None:
        nonlocal striped, accent, theme
        pitch = 28 + gap(11)  # from one line of settings to the next: its height, and a gap
        y = top()
        card((MARGIN, y), (RIGHT - MARGIN, 42 + 8 * pitch))
        ui.draw_screen_text(60, y + 30, "THE TABLE, THE GLASS, THE GAPS AND THE THEME", 11, theme.muted)

        def line(number: int, name: str) -> float:
            """Write the name of setting ``number``; return the middle of its line."""
            middle = y + 42 + (number + 0.5) * pitch
            ui.draw_screen_text(60, middle + 5, name, 13, theme.ink)
            return middle

        def slide(number: int, name: str, slider: Slider, unit: str) -> None:
            """A setting chosen with a slider, its value written after it."""
            middle = line(number, name)
            slider.start, slider.end = (330.0, middle), (610.0, middle)  # the slider follows its line
            slider.handle_color, slider.line_color, slider.hover_color = accent, theme.track, theme.ink
            ui.draw_screen_text(630, middle + 5, unit.format(slider.draw(app)), 13, theme.muted)

        # A switch is a button whose name and colour say the state. This one
        # changes every colour of the program: it picks the other theme.
        middle = line(7, "Dark theme")
        if button((330, middle - BUTTON_H / 2), (96, BUTTON_H), "on" if theme.dark else "off", on=theme.dark):
            theme = LIGHT if theme.dark else DARK
        middle = line(0, "Two backgrounds, in turn")
        if button((330, middle - BUTTON_H / 2), (96, BUTTON_H), "on" if striped else "off", on=striped):
            striped = not striped
        slide(1, "Height of a line", row_height, "{:.0f} px")
        middle = line(2, "Accent colour")
        for index, (name, color) in enumerate((("blue", BLUE), ("teal", TEAL), ("purple", PURPLE))):
            if button((330 + (96 + gap(4)) * index, middle - BUTTON_H / 2), (96, BUTTON_H), name, on=color == accent, color=color):
                accent = color
        # The glass of the panels and of the tables — this card is one: move a slider and watch it.
        slide(3, "Glass: blur", card_blur, "{:.0f} px")
        slide(4, "Glass: tint", card_white, "{:.0%}")
        slide(5, "Background: blur", deep_blur, "{:.0f} px")
        # Every gap of the three pages, this one included.
        slide(6, "Spacing", spacing, "{:.0%}")

        # A preview: the same call as on the Devices page, with the lines that still fit.
        anchor = (MARGIN, y + 42 + 8 * pitch + gap(20))
        lines = min(int((app.size()[1] - 44 - anchor[1]) // row_height.value) - 1, 3)
        if lines >= 1:
            size = (RIGHT - MARGIN, (lines + 1) * row_height.value)
            ui.draw_table(
                anchor, size, [(d["name"], d["room"], d["state"]) for d in devices[:lines]],
                theme.ink, **table_look(), header=("NAME", "ROOM", "STATE"), row_height=row_height.value,
            )  # fmt: skip
            ui.draw_screen_rect_outline(anchor, size, theme.border, 0, 1)

    draw_page = {"Overview": overview, "Devices": device_list, "Settings": settings}

    def draw_frame() -> None:
        nonlocal frames
        frames += 1
        width, height = app.size()

        # --- the background: a picture, then frosted glass over all of it --
        # The picture covers the window whatever its proportions: it is as
        # wide or as tall as needed, centred, and what sticks out is not seen.
        scale = max(width / picture.width, height / picture.height)
        size = (picture.width * scale, picture.height * scale)
        ui.draw_screen_image((0.5 * (width - size[0]), 0.5 * (height - size[1])), size, picture)
        if deep_blur.value > 0:
            ui.draw_screen_acrylic_rect((0, 0), (width, height), theme.glass, blur=deep_blur.value)
        else:  # no blur asked: the picture as it is, under the same light tint
            ui.draw_screen_rect((0, 0), (width, height), theme.glass)

        # --- the keyboard first, so the buttons below show the page it chose
        for key in app.keys_pressed():
            if key in ("RIGHT", "TAB"):
                go_to(PAGES[(PAGES.index(page) + 1) % len(PAGES)])
            elif key == "LEFT":
                go_to(PAGES[(PAGES.index(page) - 1) % len(PAGES)])
        # --- the three buttons: the page shown wears the accent colour ----
        # They belong together, so they sit in one rounded box of glass: a
        # single control with three positions. Inside it only the page shown
        # is filled; the two others are bare names that light up under the cursor.
        inside = len(PAGES) * BUTTON_W + (len(PAGES) - 1) * gap(4)
        card((MARGIN, MARGIN), (inside + 2 * BAR_PAD, BUTTON_H + 2 * BAR_PAD))
        for index, name in enumerate(PAGES):
            anchor = (MARGIN + BAR_PAD + (BUTTON_W + gap(4)) * index, MARGIN + BAR_PAD)
            if name == page:
                clicked = button(anchor, (BUTTON_W, BUTTON_H), name, on=True)
            else:
                clicked = ui.draw_button(
                    anchor, (BUTTON_W, BUTTON_H), name, (*theme.ink, 0.0), theme.hover, theme.ink, hover_text_color=theme.ink
                )
            if clicked:
                go_to(name)

        draw_page[page]()  # the page: one function

        ui.draw_screen_text(40, height - 22, f"frames drawn: {frames}      keys: Left / Right change page", 11, theme.muted)

    app.run(draw_frame)


if __name__ == "__main__":
    main()
