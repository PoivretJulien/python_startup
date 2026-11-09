"""Helpers:  python -m udp_link keygen  |  python -m udp_link ip"""

from __future__ import annotations

import secrets
import socket
import sys


def guess_local_ip() -> str:
    """LAN IP of the default-route interface (no packet is actually sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def main() -> int:
    match sys.argv[1:]:
        case ["keygen"]:
            print(secrets.token_urlsafe(32))
        case ["ip"]:
            print(guess_local_ip())
        case _:
            print("usage: python -m udp_link {keygen|ip}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
