"""CONFIRM CLIENT example -- runs on macOS, Linux and Windows.

Sends requests to apps/confirm_server.py and shows, for each one, the two
confirmations:

    request 1  led on
      sent        transport ACK: the server RECEIVED it      (12 ms)
      done        status=ok, led is on: the server PROCESSED it  (215 ms)

Run the default demo:
    macOS / Linux:  python3 -m apps.confirm_client --config link.toml
    Windows:        py -m apps.confirm_client --config link.toml

Or send your own commands ("cmd value", one per argument):
    python3 -m apps.confirm_client --config link.toml "led off" "add 5" status
"""

from __future__ import annotations

import argparse  # command-line options
import asyncio  # async runtime: futures, timeouts
import itertools  # itertools.count(): 1, 2, 3, ...
import time  # time.perf_counter(): precise clock to measure delays

# From the udp_link library:
#   UDPClient       the client (handshake, heartbeat, sends with ACKs)
#   Message         one received string
#   DeliveryError   the server didn't ACK a send after all retries
from udp_link import DeliveryError, Message, UDPClient, add_cli_args, config_from_cli

from apps._common import parse_fields, run, setup_logging

# Default demo: two good commands, one bad value, one unknown command, a status.
DEMO = ["led on", "add 5", "led blue", "reboot now", "status"]


class RequestFailed(Exception):
    """The server processed the request but reported an error."""


class Requester:
    """Send requests and wait for the server's matching 'done=<id>' reply.

    Each request gets a unique id. A Future (a placeholder for a result that
    will arrive later) is stored under that id; when the reply with the same
    id comes in, on_message() fills the Future and request() wakes up.
    """

    def __init__(self) -> None:
        self.client: UDPClient | None = None  # set once connected
        self._ids = itertools.count(1)  # request ids: 1, 2, 3, ...
        self._pending: dict[int, asyncio.Future[dict[str, str]]] = {}

    def on_message(self, msg: Message) -> None:
        """udp_link calls this for every message from the server."""
        reply = parse_fields(msg.text)  # values stay text: 'result=5' stays '5'
        if not reply.get("done", "").isdigit():
            return  # not a reply (e.g. a broadcast): ignore it here
        future = self._pending.pop(int(reply["done"]), None)
        # None: a reply that came after we stopped waiting (timeout); drop it.
        if future is not None and not future.done():
            future.set_result(reply)

    async def request(self, cmd: str, value: str = "", *, timeout: float = 5.0) -> tuple[str, float, float]:
        """Send one request; return (result, ms until ACK, ms until done).

        Raises DeliveryError   if the server never ACKed (not received),
               TimeoutError    if it was received but no reply came in time,
               RequestFailed   if the server replied with status=error.
        """
        assert self.client is not None, "connect first"
        req_id = next(self._ids)
        future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future  # register BEFORE sending: the reply can be very fast
        start = time.perf_counter()
        try:
            # 1st confirmation: returns once the server's transport ACK arrives.
            await self.client.send(f"id={req_id};cmd={cmd};value={value}")
            acked = time.perf_counter()
            # 2nd confirmation: the server's own reply, once it did the work.
            async with asyncio.timeout(timeout):
                reply = await future
        finally:
            self._pending.pop(req_id, None)  # nothing left behind on error/timeout
        done = time.perf_counter()
        if reply.get("status") != "ok":
            raise RequestFailed(reply.get("reason", "unknown error"))
        return reply.get("result", ""), (acked - start) * 1000, (done - start) * 1000


async def main() -> None:
    # --- command-line options ---------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cli_args(parser)  # --config, --server-ip, --bind-ip, --port
    parser.add_argument("commands", nargs="*", help='requests like "led on" or status (default: a demo)')
    parser.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for each reply")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    requester = Requester()
    async with UDPClient(config_from_cli(args), on_message=requester.on_message) as client:
        requester.client = client
        for n, line in enumerate(args.commands or DEMO, start=1):
            cmd, _, value = line.partition(" ")  # "led on" -> ("led", " ", "on")
            print(f"request {n}  {line}")
            try:
                result, ack_ms, done_ms = await requester.request(cmd, value.strip(), timeout=args.timeout)
            except DeliveryError:
                print("  NOT RECEIVED  no ACK after all retries (server down, wrong IP/port/secret, firewall)")
            except TimeoutError:
                print(f"  received, but NO REPLY within {args.timeout:g} s: processing state unknown")
            except RequestFailed as exc:
                print("  sent        transport ACK: the server received it")
                print(f"  FAILED      the server processed it and refused: {exc}")
            else:
                print(f"  sent        transport ACK: the server received it      ({ack_ms:.0f} ms)")
                print(f"  done        status=ok, {result}: the server processed it  ({done_ms:.0f} ms)")
        # The replies are also stacked in client.received like any message;
        # this demo only uses them through on_message, so clear the stack.
        client.drain()


# Only runs when started as a program (python3 -m apps.confirm_client).
if __name__ == "__main__":
    run(main)
