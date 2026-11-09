"""SERVER program -- run this on server_side.

It publishes a sensor reading string to every connected client at a fixed
rate, and stacks/prints the strings clients send back.

    python -m apps.sensor_server --config link.toml [--interval 1.0] [--interactive]

Replace `read_sensor()` with your real acquisition.

Sensor context when server_side is an Arduino UNO Q:
    This program runs on the Linux side of the board. On the UNO Q the sensors
    are usually wired to the MCU, and the Linux side reaches the sketch through
    the Arduino Bridge (RPC), e.g. `Bridge.call("read_temp")` inside App Lab.
    In that case, make `read_sensor()` call the Bridge and format the values
    it returns as a 'key=value;key=value' string.
"""

from __future__ import annotations

import argparse  # command-line options
import asyncio  # async runtime: tasks running side by side, non-blocking sleep
import logging
import random  # only used to simulate sensor values
import time

# From the udp_link library:
#   UDPServer             the server (listens, tracks clients, sends with ACKs)
#   Message               one received string, with sender and timestamps
#   PayloadTooLargeError  raised when a string doesn't fit in one datagram (~1100 bytes)
from udp_link import Message, PayloadTooLargeError, UDPServer, add_cli_args, config_from_cli

from apps._common import setup_logging, stdin_lines, run

log = logging.getLogger("sensor_server")


async def read_sensor() -> str:
    """Return ONE reading as a simple 'key=value;key=value' string (simulated).

    Replace the body with your real sensor reads. Keep the same output format
    so the client's parse_kv() can read it, e.g. 'temp=21.37;hum=46.2;...'.
    It is `async` so a real implementation can await slow I/O without
    freezing the rest of the program.
    """
    temp = 21.0 + random.uniform(-0.5, 0.5)  # fake temperature around 21 °C
    hum = 45.0 + random.uniform(-2.0, 2.0)  # fake humidity around 45 %
    light = random.randint(300, 800)  # fake light level
    # ts = the moment of the reading (seconds since 1970), so the client can tell its age.
    return f"temp={temp:.2f};hum={hum:.1f};light={light};ts={time.time():.3f}"


def on_client_message(msg: Message) -> None:
    # Called for every NEW string from a client (duplicates are filtered out).
    # The message is also kept in server.received (the stack).
    log.info("<- %s:%d  %r", *msg.peer, msg.text)


async def publish_loop(server: UDPServer, interval: float) -> None:
    """Read the sensor and broadcast the reading every `interval` seconds, forever."""
    loop = asyncio.get_running_loop()
    # Planned time of the next reading; planning ahead keeps an exact rhythm
    # (the time spent sending isn't added on top of `interval`).
    next_tick = loop.time()
    while True:
        reading = await read_sensor()
        if server.peers:  # at least one client connected
            # Send to every client and wait for their ACKs -> {client: delivered?}
            results = await server.broadcast(reading)
            ok = sum(results.values())  # True counts as 1: number of clients that ACKed
            log.info("-> %s  (acked by %d/%d client(s))", reading, ok, len(results))
        else:
            # Nobody to send to: the reading is dropped (shown only with --verbose).
            log.debug("no client connected, reading not sent: %s", reading)
        next_tick += interval
        await asyncio.sleep(max(0.0, next_tick - loop.time()))


async def interactive_loop(server: UDPServer) -> None:
    """Handle what the user types in the terminal (--interactive, macOS/Linux only)."""
    print("Type a line + Enter to broadcast it; '/stack' shows received messages; '/peers'.")
    async for line in stdin_lines():  # each typed line, without blocking the sends
        match line:
            case "/stack":
                # Show every string received from clients and still kept in the stack.
                for m in server.received:
                    print(f"  {m.received_at:%H:%M:%S} {m.peer[0]}:{m.peer[1]}  {m.text!r}")
                print(f"  ({len(server.received)} message(s) stacked)")
            case "/peers":
                # Addresses of the clients currently connected.
                print("  ", server.peers or "no clients")
            case _:
                # Anything else: send the typed text to every client.
                try:
                    print("  delivered:", await server.broadcast(line))
                except PayloadTooLargeError as exc:
                    print("  error:", exc)


async def main() -> None:
    # --- command-line options ---------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cli_args(parser)  # --config, --server-ip, --bind-ip, --port
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between readings")
    parser.add_argument("--interactive", action="store_true", help="also broadcast lines typed on stdin")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    # Start the server; `async with` closes it cleanly at the end.
    async with UDPServer(config_from_cli(args), on_message=on_client_message) as server:
        try:
            # A TaskGroup runs several loops at the same time. If one fails,
            # or the program is stopped, all of them are stopped together.
            async with asyncio.TaskGroup() as tg:  # runs until Ctrl+C / SIGTERM
                tg.create_task(publish_loop(server, args.interval))
                if args.interactive:
                    tg.create_task(interactive_loop(server))
        finally:
            # Always print the counters on exit (sent, delivered, retransmits, ...).
            log.info("stats: %s", server.stats)


# Only runs when started as a program (python3 -m apps.sensor_server).
if __name__ == "__main__":
    run(main)
