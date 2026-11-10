"""Example 5 — the keyboard, shown on screen.

Press keys and watch the window, all in screen space:

* **held now** — one key cap per key that is down at this moment. Hold
  several: they line up in the order you pressed them.
* **last pressed** — the latest presses, newest on the right. A key appears
  here once per press, however long you hold it.
* **typed text** — what a text field would contain. Shift, your keyboard
  layout and auto-repeat apply; Backspace deletes, Enter clears.

From the project root (after running the installer in toolchain/):

    .venv/bin/python examples/05_keyboard/main.py            # reactive (default)
    .venv/bin/python examples/05_keyboard/main.py --poll     # always redraw

The keyboard is asked three different questions, and each panel uses one:

    app.keys_down()        which keys are down *right now*        -> held now
    app.keys_pressed()     which keys went down *since last frame* -> last pressed
    app.typed()            which *characters* were produced         -> typed text

Reactive by default: a frame is drawn when a key goes down or up, and nothing
is drawn while you only hold it. Esc still quits (and you will see its key cap
for an instant).

The built-in font covers the usual ASCII letters, digits and punctuation; a
character it does not know (an accented letter, for example) is drawn as "?".
"""

from __future__ import annotations

import sys
from pathlib import Path

# The drawing library is the graphics/ folder of the project.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "graphics"))

import app3d  # noqa: E402

PURPLE = (0.58, 0.25, 0.90, 1.0)  # r, g, b, a
YELLOW = (1.00, 0.85, 0.10, 1.0)
ORANGE = (1.00, 0.55, 0.10, 1.0)
PANEL = (0.12, 0.12, 0.15, 1.0)
DIM = (0.25, 0.25, 0.31, 1.0)
LABEL = (0.85, 0.92, 1.00)
MUTED = (0.55, 0.55, 0.62)
DARK = (0.06, 0.06, 0.07)

MARGIN = 24
CAP_HEIGHT = 40  # height of a key cap
CAP_PADDING = 14  # room left and right of the name inside its cap
CAP_GAP = 8  # between two caps
TEXT_SIZE = 15.0
HISTORY = 8  # how many past presses are remembered
MAX_TEXT = 40  # characters kept in the typed line


def draw_cap(ui, x: float, y: float, name: str, fill, ink) -> float:
    """One key cap with its name centred; returns the x where the next one goes."""
    name_width = ui.measure_screen_text(name, TEXT_SIZE)
    width = max(CAP_HEIGHT, name_width + 2 * CAP_PADDING)  # never narrower than tall
    ui.draw_screen_rect((x, y), (width, CAP_HEIGHT), fill, 9)
    ui.draw_screen_text(
        x + (width - name_width) / 2,  # centred left to right
        y + (CAP_HEIGHT + TEXT_SIZE) / 2,  # the baseline: middle + half a letter
        name,
        TEXT_SIZE,
        ink,
    )
    return x + width + CAP_GAP


def main() -> int:
    app = app3d.App("example 5 — the keyboard", size=(720, 420))
    ui = app.aa

    # What the program remembers between two frames. The keys that are held
    # are NOT here: the App already knows them.
    history: list[str] = []  # the last presses, oldest first
    text = ""  # the typed line

    def draw_frame() -> None:
        nonlocal text
        width, _height = app.size()

        # --- 1. listen ----------------------------------------------------
        for name in app.keys_pressed():  # each press arrives exactly once
            history.append(name)
        del history[:-HISTORY]  # keep only the newest ones

        text += app.typed()  # characters, with Shift and the layout applied
        if app.key_pressed("BACKSPACE", repeat=True):  # repeats while held
            text = text[:-1]
        if app.key_pressed("ENTER"):
            text = ""
        text = text[-MAX_TEXT:]

        # --- 2. draw: held now ---------------------------------------------
        ui.draw_screen_text(MARGIN, 44, "held now", 13.0, MUTED)
        held = app.keys_down()
        x = MARGIN
        for name in held:
            x = draw_cap(ui, x, 58, name, YELLOW, DARK)
        if not held:
            ui.draw_screen_text(MARGIN, 84, "press any key", TEXT_SIZE, MUTED)

        # --- 3. draw: last pressed -------------------------------------------
        ui.draw_screen_text(MARGIN, 150, "last pressed", 13.0, MUTED)
        x = MARGIN
        for index, name in enumerate(history):
            newest = index == len(history) - 1
            x = draw_cap(ui, x, 164, name, PURPLE if newest else DIM, LABEL)

        # --- 4. draw: typed text ---------------------------------------------
        ui.draw_screen_text(MARGIN, 256, "typed text", 13.0, MUTED)
        field = (MARGIN, 270)
        field_size = (width - 2 * MARGIN, 52)
        ui.draw_screen_rect(field, field_size, PANEL, 10)
        baseline = field[1] + (field_size[1] + 18.0) / 2
        end = field[0] + 16 + ui.draw_screen_text(field[0] + 16, baseline, text, 18.0, LABEL)
        # The caret: a short vertical line right after the last character.
        ui.draw_screen_line((end + 4, field[1] + 12), (end + 4, field[1] + 40), ORANGE, 2)

        ui.draw_screen_text(
            MARGIN, 356, "Backspace deletes   Enter clears   Esc quits", 12.0, MUTED
        )
        shift = "yes" if app.key_down("SHIFT") else "no"  # left or right
        ui.draw_screen_text(
            MARGIN, 384, f"{len(held)} key(s) held   shift: {shift}", 12.0, MUTED
        )

    app.run(draw_frame)
    return 0


if __name__ == "__main__":
    sys.exit(main())
