"""CLIENT program -- run this on the client side.

Connects to the server, stacks every received string in `client.received`
and processes the stack in batches (here: parse 'key=value' and average).

    python -m apps.pc_client --config link.toml [--batch-interval 5] [--interactive]
"""

from __future__ import annotations

import argparse  # command-line options
import asyncio  # async runtime: tasks running side by side, non-blocking sleep
import logging
from statistics import fmean  # average of a list of floats

# From the udp_link library:
#   UDPClient             the client (handshake, heartbeat, sends with ACKs)
#   Message               one received string, with sender and timestamps
#   DeliveryError         the server didn't ACK a send after all retries
#   PayloadTooLargeError  the string doesn't fit in one datagram (~1100 bytes)
from udp_link import DeliveryError, Message, PayloadTooLargeError, UDPClient, add_cli_args, config_from_cli

from apps._common import parse_kv, setup_logging, stdin_lines, run

log = logging.getLogger("pc_client")


def process_batch(batch: list[Message]) -> None:
    """YOUR processing goes here. `batch` is the drained stack, oldest first.

    This example parses each 'temp=..;hum=..;light=..' reading and logs the
    average / min / max of each value over the batch.
    """
    readings = [parse_kv(m.text) for m in batch]  # each string -> dict
    summary = {}
    for key in ("temp", "hum", "light"):
        # Collect this value from every reading that has it as a number.
        values = [r[key] for r in readings if isinstance(r.get(key), float)]
        if values:
            summary[key] = f"avg={fmean(values):.2f} min={min(values):.2f} max={max(values):.2f}"
    # Messages that aren't 'key=value' at all (e.g. text typed on the server)
    # give an empty dict: list them separately so they aren't lost.
    others = [m.text for m, r in zip(batch, readings, strict=True) if not r]
    log.info("processed %d message(s): %s", len(batch), summary or "-")
    for text in others:
        log.info("   text message: %r", text)


async def batch_loop(client: UDPClient, every: float) -> None:
    """Process stacked messages in batches, at most once every `every` seconds.

    Event-driven, so it costs nothing while the link is quiet: instead of
    waking up every `every` seconds to look at an empty stack, it sleeps until
    the first message of a batch arrives, waits `every` seconds so that the
    following messages pile up, then processes them all together.
    """
    while True:
        # Sleeps (no CPU, no timer) until a message arrives; recv() takes it off the stack.
        first = await client.recv()
        # Let the rest of the batch arrive.
        await asyncio.sleep(every)
        # drain() returns all stacked messages and empties the stack, so each
        # message is processed exactly once.
        process_batch([first, *client.drain()])


async def interactive_loop(client: UDPClient) -> None:
    """Handle what the user types in the terminal (--interactive, macOS/Linux only)."""
    print("Type a line + Enter to send it to the server; '/stack' shows pending messages.")
    async for line in stdin_lines():  # each typed line, without blocking the receiving
        if line == "/stack":
            # Show the messages waiting for the next batch_loop round.
            for m in client.received:
                print(f"  {m.received_at:%H:%M:%S}  {m.text!r}")
            print(f"  ({len(client.received)} message(s) stacked)")
            continue
        # Anything else: send it to the server and wait for its ACK.
        try:
            await client.send(line)
            print("  ACKed by server")
        except (DeliveryError, PayloadTooLargeError) as exc:
            print("  not delivered:", exc)


async def main() -> None:
    # --- command-line options ---------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cli_args(parser)  # --config, --server-ip, --bind-ip, --port
    parser.add_argument("--batch-interval", type=float, default=5.0, help="seconds to collect a batch after its first message")
    parser.add_argument("--interactive", action="store_true", help="send lines typed on stdin to the server")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    def on_message(msg: Message) -> None:
        # Called immediately for every new message (it is ALSO added to the
        # stack). Use it for real-time reactions; here it only logs with -v.
        log.debug("<- %r", msg.text)  # real-time hook; the stack is processed in batch_loop

    # Connect (waits for the server if needed); `async with` disconnects cleanly at the end.
    async with UDPClient(config_from_cli(args), on_message=on_message) as client:
        await client.send("hello from pc_client")  # appears in the server's log
        # Run the batch processing and, optionally, the keyboard loop side by side.
        async with asyncio.TaskGroup() as tg:
            tg.create_task(batch_loop(client, args.batch_interval))
            if args.interactive:
                tg.create_task(interactive_loop(client))


# Only runs when started as a program (python3 -m apps.pc_client).
if __name__ == "__main__":
    run(main)
