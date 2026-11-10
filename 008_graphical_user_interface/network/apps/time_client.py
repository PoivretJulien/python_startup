"""TIME CLIENT test program -- runs on macOS, Linux and Windows.

Connects to the time server, prints every time message it receives and keeps
them all in a list.

macOS / Linux:  python3 -m apps.time_client --config link.toml
Windows:        py -m apps.time_client --config link.toml
"""

from __future__ import annotations

import argparse  # command-line options (--config, ...)

# UDPClient: connects to the server (handshake + heartbeat) and sends with ACKs.
from udp_link import UDPClient, add_cli_args, config_from_cli

from apps._common import parse_kv, run, setup_logging


async def main() -> None:
    # --- command-line options ---------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cli_args(parser)  # --config, --server-ip, --bind-ip, --port
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    history: list[str] = []  # every message received, oldest first

    # `async with` connects to the server (and waits for it if it isn't up
    # yet), then closes the connection cleanly when the block exits.
    async with UDPClient(config_from_cli(args)) as client:
        # A first message to the server; it appears in the server's log.
        # send() waits until the server ACKs it.
        await client.send("hello from time_client")

        # Wait for incoming messages, one at a time, forever (until Ctrl+C).
        async for msg in client.messages():
            history.append(msg.text)  # keep it
            data = parse_kv(msg.text)  # 'time=...;elapsed=63.0;tick=21' -> dict
            if "time" in data and "elapsed" in data:  # a time message
                # flush=True prints immediately, even when output goes to a file or pipe.
                print(f"server time {data['time']}  |  elapsed {data['elapsed']:.1f} s  |  "
                      f"{len(history)} message(s) received", flush=True)
            else:  # any other string the server sent
                print(f"message: {msg.text!r}", flush=True)


# Only runs when started as a program (python3 -m apps.time_client).
if __name__ == "__main__":
    run(main)
