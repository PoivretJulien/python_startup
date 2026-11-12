"""
    DHT PC CLIENT -- runs on the PC (macOS, Linux, Windows).

Connects to apps/sensor_server.py on the Arduino UNO Q and receives the DHT
readings it broadcasts every 2 s:

    temp=23.4;hum=56.0;seq=12;ts=1731340000.123
    error=read;stage=3;idle=1;release_us=12;seq=13;ts=1731340002.125

Each reading is printed as it arrives; every --batch-interval seconds the
stacked readings are summarised (avg / min / max). With --csv every reading is
also appended to a CSV file.

    python3 -m apps.pc_client --config link.toml
    python3 -m apps.pc_client --config link.toml --batch-interval 60 --csv readings.csv
    python3 -m apps.pc_client --config link.toml --interactive      # type 'get' for the latest value
"""

from __future__ import annotations

import argparse  # command-line options
import asyncio  # async runtime: tasks running side by side, non-blocking sleep
import csv
import logging
import sys
import threading  # the UDP client runs in its own thread, next to the window
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from statistics import fmean
import math

# From the udp_link library:
#   UDPClient             the client (handshake, heartbeat, sends with ACKs)
#   Message               one received string, with sender and timestamps
#   DeliveryError         the server didn't ACK a send after all retries
#   PayloadTooLargeError  the string doesn't fit in one datagram (~1100 bytes)
from udp_link import (
    DeliveryError,
    Message,
    PayloadTooLargeError,
    UDPClient,
    add_cli_args,
    config_from_cli,
)

from apps._common import parse_kv, setup_logging, stdin_lines

# The drawing library is the graphics/ folder of the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graphics"))

import app3d  # noqa: E402

log = logging.getLogger("pc_client")

CSV_FIELDS = ["time", "temp", "hum", "error_stage", "seq"]

LABEL = (0.92, 0.92, 0.36)  # r, g, b
MUTED = (0.55, 0.55, 0.62)


class Receiver:
    """Real-time handling of each message: print it, spot gaps, log to CSV."""

    def __init__(
        self, csv_path: Path | None, on_reading: Callable[[], None] | None = None
    ) -> None:
        self.last_seq: int | None = None
        # The last good reading, (when, temp, hum, seq): written here on the
        # network thread, read by draw_frame() on the window thread.
        self.latest: tuple[datetime, float, float, int | None] | None = None
        self.on_reading = on_reading  # called after each new reading
        self.csv_path = csv_path
        if csv_path is not None and not csv_path.exists():
            with csv_path.open("w", newline="") as fh:
                csv.writer(fh).writerow(CSV_FIELDS)

    def on_message(self, msg: Message) -> None:
        data = parse_kv(msg.text)
        if "temp" not in data and "error" not in data:
            log.info("<- text: %r", msg.text)  # anything else the server sends
            return

        ts = data.get("ts")
        when = datetime.fromtimestamp(ts) if isinstance(ts, float) else msg.received_at
        age = f"{time.time() - ts:.2f} s" if isinstance(ts, float) else "?"

        seq = int(data["seq"]) if isinstance(data.get("seq"), float) else None
        if seq is not None and self.last_seq is not None:
            if seq <= self.last_seq:
                # Counter went back: the board app restarted.
                log.info("server restarted (seq %d -> %d)", self.last_seq, seq)
            elif seq > self.last_seq + 1:
                log.warning(
                    "missed %d reading(s) (seq %d -> %d)",
                    seq - self.last_seq - 1,
                    self.last_seq,
                    seq,
                )
        self.last_seq = seq if seq is not None else self.last_seq

        if "error" in data:

            def as_int(key: str):
                value = data.get(key)
                return int(value) if isinstance(value, float) else "?"

            stage = as_int("stage")
            log.warning(
                "sensor read error at %s: stage=%s idle=%s release_us=%s",
                f"{when:%H:%M:%S}",
                stage,
                as_int("idle"),
                as_int("release_us"),
            )
            self._write_csv(when, None, None, stage, seq)
        else:
            temp, hum = data.get("temp"), data.get("hum")
            print(
                f"{when:%H:%M:%S}  {temp:5.1f} °C  {hum:5.1f} %RH   (seq {seq}, age {age})",
                flush=True,
            )
            self._write_csv(when, temp, hum, None, seq)
            self.latest = (when, temp, hum, seq)
            if self.on_reading is not None:
                self.on_reading()

    def _write_csv(self, when: datetime, temp, hum, stage, seq) -> None:
        if self.csv_path is None:
            return
        with self.csv_path.open("a", newline="") as fh:
            csv.writer(fh).writerow(
                [when.isoformat(timespec="seconds"), temp, hum, stage, seq]
            )


def summarise(batch: list[Message]) -> None:
    """Batch processing of the drained stack (oldest first): avg / min / max."""
    readings = [parse_kv(m.text) for m in batch]
    errors = sum(1 for r in readings if "error" in r)
    parts = []
    for key, unit in (("temp", "°C"), ("hum", "%RH")):
        values = [r[key] for r in readings if isinstance(r.get(key), float)]
        if values:
            parts.append(
                f"{key} avg={fmean(values):.1f} min={min(values):.1f} max={max(values):.1f} {unit}"
            )
    log.info(
        "summary of %d message(s): %s%s",
        len(batch),
        " | ".join(parts) or "-",
        f" | {errors} read error(s)" if errors else "",
    )


async def batch_loop(client: UDPClient, every: float) -> None:
    """Sleep until a message arrives, let the batch pile up, then summarise it."""
    while True:
        first = await client.recv()
        await asyncio.sleep(every)
        summarise([first, *client.drain()])


async def interactive_loop(client: UDPClient) -> None:
    """Send typed lines to the server (--interactive, macOS/Linux only)."""
    print(
        "Type 'get' + Enter for the latest reading, or any text to send it to the board."
    )
    async for line in stdin_lines():
        try:
            await client.send(line)
        except (DeliveryError, PayloadTooLargeError) as exc:
            print("  not delivered:", exc)


async def network(args: argparse.Namespace, receiver: Receiver) -> None:
    """The UDP side: connect, then receive until cancelled."""
    config = config_from_cli(args)
    log.info("connecting to sensor server %s:%d ...", config.server_ip, config.port)
    # Connects (waits for the board if it isn't up yet); disconnects cleanly at the end.
    async with UDPClient(config, on_message=receiver.on_message) as client:
        log.info("connected")
        await client.send("hello from pc_client")
        await client.send("get")  # ask for the latest reading right away
        async with asyncio.TaskGroup() as tg:
            tg.create_task(batch_loop(client, args.batch_interval))
            if args.interactive:
                tg.create_task(interactive_loop(client))


ANGLE = 0

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_cli_args(parser)  # --config, --server-ip, --bind-ip, --port
    parser.add_argument(
        "--batch-interval",
        type=float,
        default=30.0,
        help="seconds per summary (default 30)",
    )
    parser.add_argument(
        "--csv", type=Path, help="append every reading to this CSV file"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="send lines typed on stdin to the board",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    # The window owns the main thread. App parses the command line itself
    # (it adds its own --poll, --refresh ... options to our parser).
    app = app3d.App(
        "Sensor reading from an Arduino UNO Q", size=(560, 500), parser=parser
    )
    args = app.args
    setup_logging(args.verbose)

    # Each new reading asks the window for one frame (safe from any thread).
    receiver = Receiver(args.csv, on_reading=app.request_redraw)

    # The UDP client: its own event loop, in its own thread.
    loop = asyncio.new_event_loop()
    task = loop.create_task(network(args, receiver))

    window_closed = False  # set once app.run() has returned

    def network_thread() -> None:
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass  # cancelled by us when the window closes: the normal way out
        except Exception:
            log.exception("network stopped")
        finally:
            if not window_closed:
                app.quit()  # the network stopped on its own: close the window too

    thread = threading.Thread(target=network_thread, name="udp-client", daemon=True)
    thread.start()
    scalar = 48
    step = (math.pi/48)*4

    def draw_frame() -> None:
        global ANGLE
        ui = app.aa
        reading = receiver.latest
        if reading is None:
            ui.draw_screen_text(40, 100, "waiting for the board ...", 18.0, MUTED)
            return
        cx, cy = app.cursor()
        ui.draw_screen_text(19, 30, f"cursor: {cx:.1f},{cy:.1f}")
        ui.draw_screen_rect_outline((18, 50), (230, 160), (0, 0.65, 0.61, 1), 15, 5)
        when, temp, hum, seq = reading
        ui.draw_screen_text(40, 120, f"{temp:.1f} °C", 48.0, LABEL)
        ui.draw_screen_text(40, 190, f"{hum:.1f} %RH", 32.0, LABEL)
        ui.draw_screen_text(
            40, 480, f"{when:%H:%M:%S}   seq {seq}   Esc quits", 12.0, MUTED
        )
        
        ui.draw_screen_circle_outline((401, 132), 65, (1, 1, 0), 3)
        wx, wy = app.size()
        
        # print the window current size
        ui.draw_screen_text(210, 30, f"screen size: ({wx},{wy})", 12, (0.8, 0.0, 0.6))
        # import an image.
        ui.draw_screen_image((10, 217), (249, 248), "assets/rooster.jpg")

        # color of the rectangle.
        green = (0, 1, 0, 1)

        # Apply the right multiplayer from input keys.
        if app.key_pressed("j",True):
            if app.key_down("LEFT SHIFT"):
                ANGLE-=step*0.2
            else:
                ANGLE-=step
        if app.key_pressed("k",True):
            if app.key_down("LEFT SHIFT"):
                ANGLE+=step*0.2
            else:
                ANGLE+=step

        # Compute the triangle position and orientation
        dir = (math.cos(ANGLE), math.sin(ANGLE))  # direction
        x =  401 + (dir[0] * scalar)
        y =  132 + (dir[1] * scalar)   
        ui.draw_screen_triangle((x,y), 15, 15, dir, green)
        ui.draw_screen_text(328, 215,"hold shift (to fine tune)", 12, (0.8, 0.0, 0.6))
        ui.draw_screen_text(375, 138,"(-)j/(+)k", 12, (0.8, 0.0, 0.6))

        # make an amber semi transparent frame.
        ui.draw_screen_inverted_rect((25,25),(wx-50,wy-50),(1,0.53,0,0.25),10)
        ui.draw_screen_acrylic_rect((161,250),(118,198),(0.6,0.2,0.8,0.2),18,15)
        ui.draw_screen_text(172, 276,"Privacy: ", 12, (0.95, 0.78, 0.12))

    try:
        app.run(draw_frame)  # blocks until Esc, the close button or Ctrl+C
    finally:
        window_closed = True
        # Cancel the client so it leaves its `async with` and says goodbye (BYE).
        if not loop.is_closed():
            loop.call_soon_threadsafe(task.cancel)
        thread.join(timeout=5)
        if not thread.is_alive():
            loop.close()


if __name__ == "__main__":
    main()
