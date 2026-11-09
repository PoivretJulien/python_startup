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

import argparse
import asyncio
import logging
import random
import time

from udp_link import Message, PayloadTooLargeError, UDPServer, add_cli_args, config_from_cli

from apps._common import setup_logging, stdin_lines, run

log = logging.getLogger("sensor_server")


async def read_sensor() -> str:
    """Return ONE reading as a simple 'key=value;key=value' string (simulated)."""
    temp = 21.0 + random.uniform(-0.5, 0.5)
    hum = 45.0 + random.uniform(-2.0, 2.0)
    light = random.randint(300, 800)
    return f"temp={temp:.2f};hum={hum:.1f};light={light};ts={time.time():.3f}"


def on_client_message(msg: Message) -> None:
    # Called for every NEW string from a client (duplicates are filtered out).
    # The message is also kept in server.received (the stack).
    log.info("<- %s:%d  %r", *msg.peer, msg.text)


async def publish_loop(server: UDPServer, interval: float) -> None:
    loop = asyncio.get_running_loop()
    next_tick = loop.time()
    while True:
        reading = await read_sensor()
        if server.peers:
            results = await server.broadcast(reading)
            ok = sum(results.values())
            log.info("-> %s  (acked by %d/%d client(s))", reading, ok, len(results))
        else:
            log.debug("no client connected, reading not sent: %s", reading)
        next_tick += interval
        await asyncio.sleep(max(0.0, next_tick - loop.time()))


async def interactive_loop(server: UDPServer) -> None:
    print("Type a line + Enter to broadcast it; '/stack' shows received messages; '/peers'.")
    async for line in stdin_lines():
        match line:
            case "/stack":
                for m in server.received:
                    print(f"  {m.received_at:%H:%M:%S} {m.peer[0]}:{m.peer[1]}  {m.text!r}")
                print(f"  ({len(server.received)} message(s) stacked)")
            case "/peers":
                print("  ", server.peers or "no clients")
            case _:
                try:
                    print("  delivered:", await server.broadcast(line))
                except PayloadTooLargeError as exc:
                    print("  error:", exc)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cli_args(parser)
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between readings")
    parser.add_argument("--interactive", action="store_true", help="also broadcast lines typed on stdin")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    async with UDPServer(config_from_cli(args), on_message=on_client_message) as server:
        try:
            async with asyncio.TaskGroup() as tg:  # runs until Ctrl+C / SIGTERM
                tg.create_task(publish_loop(server, args.interval))
                if args.interactive:
                    tg.create_task(interactive_loop(server))
        finally:
            log.info("stats: %s", server.stats)


if __name__ == "__main__":
    run(main)
