"""Video settings. Addressing and the secret come from udp_link's LinkConfig
(same link.toml / env vars); this file only adds video-specific knobs, read
from an optional [video] table in the same TOML file."""

from __future__ import annotations

import os
import socket
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

DEFAULT_VIDEO_PORT = 50556


@dataclass(frozen=True, slots=True)
class VideoConfig:
    port: int = DEFAULT_VIDEO_PORT
    subscribe_interval: float = 1.0  # client SUB keep-alive period (s)
    subscriber_ttl: float = 4.0  # server stops streaming to silent clients (s)
    max_subscribers: int = 4  # each subscriber costs full upload bandwidth
    max_frame_bytes: int = 1_000_000  # reject frames larger than this
    max_partial_frames: int = 4  # frames being reassembled at once (client)
    buffer_frames: int = 30  # client keeps the N most recent frames
    socket_buffer: int = 4 * 1024 * 1024  # SO_SNDBUF / SO_RCVBUF request
    max_write_buffer: int = 2 * 1024 * 1024  # server skips frames above this backlog

    def __post_init__(self) -> None:
        if not 0 <= self.port <= 65535:
            raise ValueError(f"port out of range: {self.port}")
        for f in fields(self):
            if f.name != "port" and getattr(self, f.name) <= 0:
                raise ValueError(f"{f.name} must be > 0")

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None, **overrides: Any) -> VideoConfig:
        values: dict[str, Any] = {}
        path = path or os.environ.get("UDP_LINK_CONFIG")
        if path:
            with Path(path).open("rb") as fh:
                values.update(tomllib.load(fh).get("video", {}))
        if (port := os.environ.get("UDP_LINK_VIDEO_PORT")) is not None:
            values["port"] = int(port)
        values.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        if unknown := set(values) - known:
            raise ValueError(f"unknown [video] keys: {sorted(unknown)}")
        return cls(**values)


def make_udp_socket(bind: tuple[str, int], buffer_size: int) -> socket.socket:
    """IPv4 UDP socket with enlarged kernel buffers (video sends bursts)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, opt, buffer_size)
        except OSError:
            pass  # OS limit (e.g. net.core.rmem_max); keep the default
    sock.bind(bind)
    sock.setblocking(False)
    return sock
