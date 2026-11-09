"""CLIENT program -- run this on your PC / local Python app.

Connects to the server, stacks every received string in `client.received`
and processes the stack in batches (here: parse 'key=value' and average).

    python -m apps.pc_client --config link.toml [--batch-interval 5] [--interactive]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from statistics import fmean

from udp_link import DeliveryError, Message, PayloadTooLargeError, UDPClient, add_cli_args, config_from_cli

from apps._common import parse_kv, setup_logging, stdin_lines, run

log = logging.getLogger("pc_client")


def process_batch(batch: list[Message]) -> None:
    """YOUR processing goes here. `batch` is the drained stack, oldest first."""
    readings = [parse_kv(m.text) for m in batch]
    summary = {}
    for key in ("temp", "hum", "light"):
        values = [r[key] for r in readings if isinstance(r.get(key), float)]
        if values:
            summary[key] = f"avg={fmean(values):.2f} min={min(values):.2f} max={max(values):.2f}"
    others = [m.text for m, r in zip(batch, readings, strict=True) if not r]
    log.info("processed %d message(s): %s", len(batch), summary or "-")
    for text in others:
        log.info("   text message: %r", text)


async def batch_loop(client: UDPClient, every: float) -> None:
    while True:
        await asyncio.sleep(every)
        if batch := client.drain():
            process_batch(batch)


async def interactive_loop(client: UDPClient) -> None:
    print("Type a line + Enter to send it to the server; '/stack' shows pending messages.")
    async for line in stdin_lines():
        if line == "/stack":
            for m in client.received:
                print(f"  {m.received_at:%H:%M:%S}  {m.text!r}")
            print(f"  ({len(client.received)} message(s) stacked)")
            continue
        try:
            await client.send(line)
            print("  ACKed by server")
        except (DeliveryError, PayloadTooLargeError) as exc:
            print("  not delivered:", exc)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cli_args(parser)
    parser.add_argument("--batch-interval", type=float, default=5.0, help="seconds between stack processing")
    parser.add_argument("--interactive", action="store_true", help="send lines typed on stdin to the server")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    def on_message(msg: Message) -> None:
        log.debug("<- %r", msg.text)  # real-time hook; the stack is processed in batch_loop

    async with UDPClient(config_from_cli(args), on_message=on_message) as client:
        await client.send("hello from pc_client")
        async with asyncio.TaskGroup() as tg:
            tg.create_task(batch_loop(client, args.batch_interval))
            if args.interactive:
                tg.create_task(interactive_loop(client))


if __name__ == "__main__":
    run(main)
