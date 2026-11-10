"""Example 2 — drag a circle in one window, watch it move in the other (UDP).

Two windows, side by side:

* **side A** owns the circle. Click it and drag: it follows the cursor and
  turns from purple to yellow while held. Every change is sent as one small
  string, e.g. ``x=212.0;y=148.5;drag=1``.
* **side B** draws what it receives: the same position, the same colour.

From the project root (after running the installer in toolchain/):

    .venv/bin/python examples/02_udp_drag/main.py            # both windows, reactive
    .venv/bin/python examples/02_udp_drag/main.py --poll     # both, always redraw
    .venv/bin/python examples/02_udp_drag/main.py --side a   # only side A
    .venv/bin/python examples/02_udp_drag/main.py --side b   # only side B

Reactive is the default on both sides: side A redraws when the mouse does
something, side B when a message arrives (the link thread wakes the sleeping
window with ``app.request_redraw()``). Idle windows draw nothing.

------------------------------------------------------------------------------
How to configure link.toml
------------------------------------------------------------------------------
Both sides read the same file: ``network/link.toml`` (or the one
given with ``--config``). Side A is the *server* (it listens and broadcasts the
position), side B is the *client* (it connects to ``server_ip``).

1. Same machine — works out of the box. This is the file as shipped::

       [link]
       server_ip = "127.0.0.1"    # side B looks for side A on this machine
       bind_ip   = "0.0.0.0"      # side A listens on every interface
       port      = 50555
       secret    = "..."          # any shared text of 16+ characters

2. Two machines on the same local network — change one line, ``server_ip``:

   a. on the machine that runs side A, find its LAN address::

          cd network && python3 -m udp_link ip      # e.g. 192.168.1.50

   b. put it in link.toml::

          server_ip = "192.168.1.50"   # the LAN IP of the side A machine

      (a fresh secret is a good idea too: ``python3 -m udp_link keygen``)

   c. copy the **same** link.toml to both machines (copying the whole
      project folder is the simplest), then::

          machine A:  .venv/bin/python examples/02_udp_drag/main.py --side a
          machine B:  .venv/bin/python examples/02_udp_drag/main.py --side b

   If side B stays on "waiting for side A": the firewall of machine A must let
   UDP port 50555 in, both files must hold the same secret, and the two clocks
   must agree within two minutes (or set ``max_clock_skew = 0`` in both files).

``link.example.toml`` lists every optional key. Environment variables
(``UDP_LINK_SERVER_IP``, ``UDP_LINK_PORT``, ``UDP_LINK_SECRET``) override the
file, which is handy for a quick test without editing it.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "graphics"))  # the drawing library
sys.path.insert(0, str(ROOT / "network"))  # the udp library

DEFAULT_CONFIG = ROOT / "network" / "link.toml"

PURPLE = (0.58, 0.25, 0.90, 1.0)
YELLOW = (1.00, 0.85, 0.10, 1.0)
LABEL = (0.85, 0.92, 1.00)
PANEL = (0.12, 0.12, 0.15, 1.0)

WINDOW = (560, 420)
GAP = 24  # pixels between the two windows
TOP_LEFT = (80, 140)  # where side A opens on the desktop
RADIUS = 46
START = (WINDOW[0] / 2, WINDOW[1] / 2)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drag a circle on side A, mirrored on side B over UDP.",
    )
    parser.add_argument(
        "--side",
        choices=("a", "b", "both"),
        default="both",
        help="a = the window you drag in, b = the mirror, both = open the two (default)",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="link.toml to use (default: network/link.toml)",
    )
    return parser


def open_side(side: str, parser: argparse.ArgumentParser):
    """The window of one side, placed so A and B sit next to each other."""
    import app3d

    x = TOP_LEFT[0] + (WINDOW[0] + GAP if side == "b" else 0)
    title = "side A — drag the circle" if side == "a" else "side B — mirror of side A"
    return app3d.App(title, size=WINDOW, position=(x, TOP_LEFT[1]), parser=parser)


def draw_scene(app, title: str, position, held: bool, status: str) -> None:
    """What both sides draw: a header, the circle, a status line."""
    ui = app.aa
    width, height = app.size()
    ui.draw_screen_rect((12, 12), (width - 24, 40), PANEL, 10)
    ui.draw_screen_text(26, 38, title, 14.0, LABEL)
    if position is not None:
        ui.draw_screen_circle(position, RADIUS, YELLOW if held else PURPLE)
    ui.draw_screen_text(26, height - 20, status, 12.0, LABEL)


# ----------------------------------------------------------------- side A
def run_side_a(parser: argparse.ArgumentParser) -> None:
    """Owns the circle: drag it with the mouse, send every change."""
    from udp_link.bridge import LinkThread

    app = open_side("a", parser)
    ui = app.aa

    position = list(START)  # centre of the circle, window pixels
    grab = (0.0, 0.0)  # where inside the circle it was grabbed
    dragging = False
    was_down = False
    sent: str | None = None  # the last string handed to the link

    def on_text(_text: str) -> None:
        # Side B says hello when it connects: forget what was sent, so the
        # next frame sends the current state again, and ask for that frame.
        nonlocal sent
        sent = None
        app.request_redraw()

    link = LinkThread("server", app.args.config, on_text=on_text).start()

    def draw_frame() -> None:
        nonlocal dragging, was_down, grab, sent

        cursor = app.cursor()
        down = app.mouse_down()

        if down and not was_down and ui.in_screen_circle(cursor, position, RADIUS):
            # The click landed on the circle: remember where, so the circle
            # does not jump its centre under the cursor.
            dragging = True
            grab = (position[0] - cursor[0], position[1] - cursor[1])
        if not down:
            dragging = False
        was_down = down

        if dragging:
            width, height = app.size()
            position[0] = min(max(cursor[0] + grab[0], 0.0), width)
            position[1] = min(max(cursor[1] + grab[1], 0.0), height)

        # One short string carries the whole state; send it only when it changed.
        text = f"x={position[0]:.1f};y={position[1]:.1f};drag={int(dragging)}"
        if text != sent:
            link.send(text)
            sent = text

        status = f"sending on udp port {link.config.port}"
        if link.error is not None:
            status = f"link error: {link.error}"
        draw_scene(app, "SIDE A   click the circle and drag it", position, dragging, status)

    try:
        app.run(draw_frame)
    finally:
        link.close()


# ----------------------------------------------------------------- side B
def run_side_b(parser: argparse.ArgumentParser) -> None:
    """Draws what side A sends. It never reads its own mouse."""
    from apps._common import parse_kv  # 'x=1;y=2' -> {'x': 1.0, 'y': 2.0}
    from udp_link.bridge import LinkThread

    app = open_side("b", parser)

    # The latest state from side A. One tuple, replaced whole, so the drawing
    # thread always reads a consistent (x, y, held).
    remote: tuple[float, float, bool] | None = None

    def on_text(text: str) -> None:
        # Runs in the link thread: store the value, wake the window, return.
        nonlocal remote
        values = parse_kv(text)
        try:
            remote = (float(values["x"]), float(values["y"]), bool(values.get("drag", 0.0)))
        except (KeyError, TypeError, ValueError):
            return  # not a position: ignore it
        app.request_redraw()

    link = LinkThread("client", app.args.config, on_text=on_text).start()
    link.send("hello")  # queued until connected; side A answers with its state

    def draw_frame() -> None:
        target = f"{link.config.server_ip}:{link.config.port}"
        if link.error is not None:
            status = f"link error: {link.error}"
        elif remote is None:
            status = f"waiting for side A at {target} ..."
        else:
            status = f"receiving from {target}   ({remote[0]:.0f}, {remote[1]:.0f})"
        position = remote[:2] if remote is not None else None
        held = remote[2] if remote is not None else False
        draw_scene(app, "SIDE B   mirror of side A", position, held, status)

    try:
        app.run(draw_frame)
    finally:
        link.close()


# ------------------------------------------------------------------- both
def run_both(argv: list[str]) -> int:
    """Open side A and side B as two processes, exactly as on two machines."""
    command = [sys.executable, str(Path(__file__).resolve())]
    side_a = subprocess.Popen([*command, *argv, "--side", "a"])
    time.sleep(0.4)  # let side A start listening, so side B connects at once
    side_b = subprocess.Popen([*command, *argv, "--side", "b"])
    children = (side_a, side_b)

    # Closing either window ends the demonstration: sleep until one exits.
    one_closed = threading.Event()
    for child in children:
        threading.Thread(
            target=lambda child=child: (child.wait(), one_closed.set()), daemon=True
        ).start()
    try:
        one_closed.wait()
    except KeyboardInterrupt:
        pass  # Ctrl+C reaches the two windows too: they close by themselves
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        child.wait()
    return 0


def main() -> int:
    parser = make_parser()
    side = parser.parse_known_args()[0].side  # the loop options are parsed by App
    if side == "both":
        return run_both(sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO, format=f"side {side.upper()} %(levelname)-7s %(message)s"
    )
    (run_side_a if side == "a" else run_side_b)(parser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
