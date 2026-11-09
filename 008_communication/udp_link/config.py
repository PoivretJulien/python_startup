"""Shared configuration: both sides must agree on server_ip, port and secret.

Precedence (highest first): explicit overrides / CLI flags > environment
variables > TOML file > defaults.

Environment variables:
    UDP_LINK_CONFIG     path to a TOML file (see link.example.toml)
    UDP_LINK_SERVER_IP  IP of the machine running the server (client side)
    UDP_LINK_BIND_IP    IP the server listens on (0.0.0.0 = all interfaces)
    UDP_LINK_PORT       UDP port
    UDP_LINK_SECRET     shared secret (>= 16 chars). Generate: python -m udp_link keygen
"""

from __future__ import annotations

import argparse
import logging
import os
import stat
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .protocol import derive_key

log = logging.getLogger(__name__)

DEFAULT_PORT = 50555
MIN_SECRET_LEN = 16
DEFAULT_CONFIG_NAME = "link.toml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_config_path(path: str | os.PathLike[str] | None) -> Path | None:
    """Find the config file so the programs work from any folder.

    Explicit path (or $UDP_LINK_CONFIG): used as given; if relative and not
    found in the current folder, it is also looked up in the project folder.
    No path: link.toml in the current folder, then in the project folder.
    """
    path = path or os.environ.get("UDP_LINK_CONFIG")
    if path:
        candidate = Path(path).expanduser()
        if candidate.exists() or candidate.is_absolute():
            return candidate
        if (in_project := PROJECT_ROOT / candidate).exists():
            return in_project
        raise FileNotFoundError(f"config file {path!r} not found in {Path.cwd()} or {PROJECT_ROOT}")
    for folder in (Path.cwd(), PROJECT_ROOT):
        if (candidate := folder / DEFAULT_CONFIG_NAME).exists():
            return candidate
    return None

_ENV = {
    "server_ip": ("UDP_LINK_SERVER_IP", str),
    "bind_ip": ("UDP_LINK_BIND_IP", str),
    "port": ("UDP_LINK_PORT", int),
    "secret": ("UDP_LINK_SECRET", str),
}


@dataclass(frozen=True, slots=True)
class LinkConfig:
    # --- addressing (must match on both machines) ---
    server_ip: str = "127.0.0.1"  # where the client sends to
    bind_ip: str = "0.0.0.0"  # where the server listens
    client_bind_ip: str = "0.0.0.0"  # client local interface (port is ephemeral)
    port: int = DEFAULT_PORT
    secret: str = field(default="", repr=False)

    # --- reliability ---
    ack_timeout: float = 0.3  # first retransmission timeout (s)
    max_retries: int = 5  # retransmissions after the first send
    max_backoff: float = 2.0  # cap for exponential backoff (s)
    max_in_flight: int = 64  # concurrent un-ACKed sends per endpoint

    # --- liveness ---
    heartbeat_interval: float = 5.0  # client PING period (s)
    peer_ttl: float = 20.0  # server forgets silent clients after this (s)

    # --- security / resource limits ---
    max_clock_skew: float = 120.0  # replay window (s); 0 disables (no NTP on LAN)
    max_peers: int = 32
    inbox_limit: int = 10_000  # max stacked messages; oldest dropped beyond
    allowed_hosts: frozenset[str] = frozenset()  # empty = any authenticated host

    key: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not 0 <= self.port <= 65535:
            raise ValueError(f"port out of range: {self.port}")
        if len(self.secret) < MIN_SECRET_LEN:
            raise ValueError(
                f"a shared secret of at least {MIN_SECRET_LEN} characters is required "
                "(set UDP_LINK_SECRET or 'secret' in the config file; "
                "generate one with: python -m udp_link keygen)"
            )
        for name in ("ack_timeout", "max_backoff", "heartbeat_interval", "peer_ttl"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        for name in ("max_retries", "max_clock_skew"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("max_in_flight", "max_peers", "inbox_limit"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        object.__setattr__(self, "allowed_hosts", frozenset(self.allowed_hosts))
        object.__setattr__(self, "key", derive_key(self.secret))

    def is_allowed(self, host: str) -> bool:
        return not self.allowed_hosts or host in self.allowed_hosts

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None, **overrides: Any) -> LinkConfig:
        values: dict[str, Any] = {}

        if (found := resolve_config_path(path)) is not None:
            log.debug("using config %s", found)
            values.update(_read_toml(found))

        for name, (env, conv) in _ENV.items():
            if (raw := os.environ.get(env)) is not None:
                values[name] = conv(raw)

        values.update({k: v for k, v in overrides.items() if v is not None})

        known = {f.name for f in fields(cls) if f.init}
        if unknown := set(values) - known:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**values)


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    data = data.get("link", data)
    if "secret" in data and os.name == "posix":
        mode = path.stat().st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            log.warning("%s contains the secret but is readable by others; run: chmod 600 %s", path, path)
    return data


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Common CLI flags. The secret is deliberately NOT a flag (visible in `ps`)."""
    g = parser.add_argument_group("link")
    g.add_argument("--config", help="TOML config file (default: $UDP_LINK_CONFIG)")
    g.add_argument("--server-ip", help="server IP the client connects to")
    g.add_argument("--bind-ip", help="IP the server listens on (default 0.0.0.0)")
    g.add_argument("--port", type=int, help=f"UDP port (default {DEFAULT_PORT})")


def config_from_cli(args: argparse.Namespace) -> LinkConfig:
    return LinkConfig.load(args.config, server_ip=args.server_ip, bind_ip=args.bind_ip, port=args.port)
