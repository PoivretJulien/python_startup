"""Cross-platform UDP socket creation (macOS, Linux, Windows)."""

from __future__ import annotations

import socket


def open_udp_socket(host: str, port: int, *, buffer_size: int | None = None) -> socket.socket:
    """Bound, non-blocking IPv4 UDP socket ready for loop.create_datagram_endpoint(sock=...).

    Windows: when a datagram hits a closed port, the ICMP "port unreachable"
    reply is reported as WSAECONNRESET on later reads of the *same* socket,
    which can stop asyncio's UDP transport. SIO_UDP_CONNRESET turns that off.
    It is a no-op on macOS/Linux (the attribute does not exist there).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        if hasattr(socket, "SIO_UDP_CONNRESET"):
            sock.ioctl(socket.SIO_UDP_CONNRESET, False)  # type: ignore[attr-defined]
        if buffer_size:
            for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
                try:
                    sock.setsockopt(socket.SOL_SOCKET, opt, buffer_size)
                except OSError:
                    pass  # OS limit (e.g. Linux net.core.rmem_max): keep the default
        sock.bind((host, port))
        sock.setblocking(False)
    except BaseException:
        sock.close()
        raise
    return sock
