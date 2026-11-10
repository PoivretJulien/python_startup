"""CONFIRM SERVER example -- runs on macOS, Linux and Windows.

Shows how a client can be sure a message was not only RECEIVED but also
PROCESSED by the server. Two confirmations happen for every request:

  1. transport ACK (automatic, udp_link): client.send() returns
     -> the server has the message in its stack
  2. application reply (this program): once the work is done, the server
     sends back 'done=<id>;status=ok;...' or 'done=<id>;status=error;...'
     -> the server has actually handled it

Request format (sent by apps/confirm_client.py):   id=7;cmd=led;value=on
Reply format:                                      done=7;status=ok;result=led is on

macOS / Linux:  python3 -m apps.confirm_server --config link.toml
Windows:        py -m apps.confirm_server --config link.toml
"""

from __future__ import annotations

import argparse  # command-line options
import asyncio  # async runtime
import logging

# From the udp_link library:
#   UDPServer       the server (listens, tracks clients, sends with ACKs)
#   Message         one received string, with its sender (msg.peer) and timestamps
#   DeliveryError   a reply could not be delivered (client gone)
from udp_link import DeliveryError, Message, UDPServer, add_cli_args, config_from_cli

from apps._common import parse_fields, run, setup_logging

log = logging.getLogger("confirm_server")


class CommandError(Exception):
    """A request that can't be carried out; its text goes back to the client."""


class Device:
    """The 'work' this server does. Replace it with your real actions.

    Here: a pretend LED and a counter, so results are easy to see.
    """

    def __init__(self) -> None:
        self.led = "off"
        self.counter = 0

    async def execute(self, cmd: str, value: str) -> str:
        """Carry out one command and return a short result text.

        Raises CommandError for a bad request. It is `async` so real work
        (a sensor read, an Arduino Bridge call, a file write) can be awaited.
        """
        await asyncio.sleep(0.2)  # simulated processing time
        match cmd:
            case "led":
                if value not in ("on", "off"):
                    raise CommandError(f"led value must be on or off, got {value!r}")
                self.led = value
                return f"led is {self.led}"
            case "add":
                try:
                    self.counter += int(value)
                except ValueError:
                    raise CommandError(f"add needs a whole number, got {value!r}") from None
                return f"counter is {self.counter}"
            case "status":
                return f"led is {self.led}, counter is {self.counter}"
            case _:
                raise CommandError(f"unknown command {cmd!r}")


async def handle_request(server: UDPServer, device: Device, msg: Message) -> None:
    """Process one request, then reply to the client that sent it.

    Called for every new message (see main). udp_link runs it as its own
    task, so a slow request never blocks the reception of the next ones.
    """
    # 'id=7;cmd=add;value=5' -> {'id': '7', 'cmd': 'add', 'value': '5'} (all text)
    request = parse_fields(msg.text)
    if not request.get("id", "").isdigit():
        # Not a request (e.g. a plain greeting): nothing to confirm.
        log.info("<- %s:%d  %r (no id, not confirmed)", *msg.peer, msg.text)
        return
    req_id = int(request["id"])
    cmd, value = request.get("cmd", ""), request.get("value", "")
    log.info("<- %s:%d  request %d: %s %s", *msg.peer, req_id, cmd, value)

    # Do the work. Every outcome, good or bad, gets a reply, so the client
    # never has to guess.
    try:
        result = await device.execute(cmd, value)
        reply = f"done={req_id};status=ok;result={result}"
    except CommandError as exc:
        reply = f"done={req_id};status=error;reason={exc}"
    except Exception:  # a bug in execute(): report it instead of staying silent
        log.exception("request %d failed", req_id)
        reply = f"done={req_id};status=error;reason=internal error"

    # Send the application reply to the client that asked (msg.peer).
    # send() itself waits for the client's transport ACK.
    try:
        await server.send(reply, msg.peer)
        log.info("-> %s:%d  %s", *msg.peer, reply)
    except DeliveryError:
        log.warning("could not reply to %s:%d (client gone?): %s", *msg.peer, reply)
    except ConnectionError:
        pass  # the server is shutting down


async def main() -> None:
    # --- command-line options ---------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cli_args(parser)  # --config, --server-ip, --bind-ip, --port
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    device = Device()
    # The handler needs the server to reply. The lambda looks `server` up when
    # a message arrives, by which time it exists. It returns the coroutine of
    # handle_request(), which udp_link runs as a task.
    server = UDPServer(
        config_from_cli(args),
        on_message=lambda msg: handle_request(server, device, msg),
    )

    async with server:
        log.info("waiting for requests (Ctrl+C to stop)")
        try:
            await asyncio.Event().wait()  # everything happens in on_message
        finally:
            log.info("stats: %s", server.stats)


# Only runs when started as a program (python3 -m apps.confirm_server).
if __name__ == "__main__":
    run(main)
