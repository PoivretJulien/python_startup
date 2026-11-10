"""TIME SERVER test program -- runs on macOS, Linux and Windows.

Every --interval seconds (default 3) it sends the local OS time and the time
elapsed since this server started to every connected client:

    time=2026-11-09T16:30:03+01:00;elapsed=63.0;tick=21

macOS / Linux:  python3 -m apps.time_server --config link.toml
Windows:        py -m apps.time_server --config link.toml
"""

from __future__ import annotations

import argparse  # command-line options (--config, --interval, ...)
import asyncio  # async runtime (sleeping without blocking)
import logging
import time  # time.monotonic(): a clock that only moves forward
from datetime import datetime  # wall-clock date and time

# From the udp_link library:
#   UDPServer      the server itself (listens, tracks clients, sends with ACKs)
#   Message        one received string, with its sender and timestamps
#   add_cli_args / config_from_cli   the shared --config/--port/... options
from udp_link import Message, UDPServer, add_cli_args, config_from_cli

from apps._common import run, setup_logging

# A named logger: its lines are prefixed with "time_server:".
log = logging.getLogger("time_server")

def time_message(started: float, tick: int) -> str:
    """
    Build one message, e.g. 'time=2026-11-09T16:30:03+01:00;elapsed=63.0;tick=21'.

    started  value of time.monotonic() when the server started
    tick     message counter (1, 2, 3, ...), lets the client spot gaps
    """
    # local OS time, with its UTC offset
    now = datetime.now().astimezone()
    elapsed = time.monotonic() - started  
    # isoformat(timespec='seconds') -> '2026-11-09T16:30:03+01:00'; elapsed with 1 decimal.
    return f"time={now.isoformat(timespec='seconds')};elapsed={elapsed:.1f};tick={tick}"


def on_client_message(msg: Message) -> None:
    """Called once for every new string a client sends us (duplicates are filtered out)."""
    # msg.peer is the sender's (ip, port); '*' spreads it into the two %s / %d fields.
    log.info("<- %s:%d  %r", *msg.peer, msg.text)


async def main() -> None:
    # --- command-line options ---------------------------------------------
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_cli_args(parser)  # --config, --server-ip, --bind-ip, --port
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="seconds between messages (default 3)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    started = time.monotonic()  # reference point for "elapsed since launch"
    loop = asyncio.get_running_loop()

    # `async with` starts the server (opens the UDP port) and guarantees it is
    # closed again when the block exits, including on Ctrl+C.
    async with UDPServer(config_from_cli(args), on_message=on_client_message) as server:
        # next_tick is the planned send time. Planning ahead (instead of just
        # sleeping `interval` seconds) keeps the rhythm exact: the time spent
        # sending isn't added on top, so messages don't slowly drift later.
        tick, next_tick = 0, loop.time()
        while True:  # runs until Ctrl+C / SIGTERM cancels it
            tick += 1
            text = time_message(started, tick)
            if server.peers:  # at least one client is connected
                # broadcast() sends to every client and waits for their ACKs;
                # it returns {client_address: True/False (delivered or not)}.
                results = await server.broadcast(text)
                log.info(
                    "-> %s  (acked by %d/%d client(s))",
                    text,
                    sum(results.values()),
                    len(results),
                )
            else:
                log.info("-> %s  (no client connected yet)", text)
            next_tick += args.interval
            # Sleep until the next planned tick (never a negative time).
            await asyncio.sleep(max(0.0, next_tick - loop.time()))


# This block only runs when the file is started as a program
# (python3 -m apps.time_server), not when another file imports it.
if __name__ == "__main__":
    run(main)
