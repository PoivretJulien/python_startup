"""Wire format for udp_link.

Every datagram is::

    +----------------------+--------------------------------------------+
    | HMAC-SHA256 (32 B)   | UTF-8 JSON body                            |
    +----------------------+--------------------------------------------+

Body fields (short keys keep datagrams small)::

    v  protocol version (int)
    k  kind: H=HELLO, P=PING, D=DATA, A=ACK, B=BYE
    s  sender session id (random hex, changes on every restart)
    n  sequence number (uint32). For an ACK: the sequence being acknowledged
    t  sender wall-clock timestamp (float, seconds) -> replay window
    p  payload (str), only meaningful for DATA

Security properties:
    * Authenticity / integrity: HMAC over the whole body with a shared key.
      Packets with a bad MAC are silently dropped (and never ACKed).
    * Replay protection: timestamp window + per-session sequence dedup window.
    * NOT confidential: the payload is readable on the LAN. Fine for sensor
      values; use a VPN/WireGuard or add AES-GCM if you need secrecy.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
from dataclasses import dataclass
from enum import StrEnum

VERSION = 1
MAC_SIZE = 32
# Stay below the IPv6 minimum MTU (1280) so datagrams are never fragmented.
MAX_DATAGRAM_SIZE = 1200
MAX_SEQ = 0xFFFFFFFF
MAX_SID_LEN = 32
_KDF_SALT = b"udp-link/v1"
_KDF_ITERATIONS = 100_000


class Kind(StrEnum):
    HELLO = "H"  # client -> server: register me
    PING = "P"  # keep-alive / liveness probe
    DATA = "D"  # application string
    ACK = "A"  # acknowledgment of a reliable packet
    BYE = "B"  # graceful disconnect (best effort, not retried)


class ProtocolError(ValueError):
    """Datagram is malformed or not acceptable."""


class AuthenticationError(ProtocolError):
    """HMAC check failed (wrong secret or tampered packet)."""


class StalePacketError(ProtocolError):
    """Timestamp is outside the accepted clock-skew window (replay or bad clock)."""


class PayloadTooLargeError(ProtocolError):
    """Encoded datagram would exceed MAX_DATAGRAM_SIZE."""


@dataclass(frozen=True, slots=True)
class Packet:
    kind: Kind
    sid: str
    seq: int
    ts: float
    payload: str = ""


def derive_key(secret: str) -> bytes:
    """Stretch the shared secret into a 32-byte HMAC key (done once at startup)."""
    return hashlib.pbkdf2_hmac("sha256", secret.encode("utf-8"), _KDF_SALT, _KDF_ITERATIONS)


def encode(packet: Packet, key: bytes) -> bytes:
    body = json.dumps(
        {
            "v": VERSION,
            "k": packet.kind.value,
            "s": packet.sid,
            "n": packet.seq,
            "t": round(packet.ts, 3),
            "p": packet.payload,
        },
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    size = MAC_SIZE + len(body)
    if size > MAX_DATAGRAM_SIZE:
        raise PayloadTooLargeError(
            f"datagram would be {size} bytes (max {MAX_DATAGRAM_SIZE}); "
            "shorten the string or split it"
        )
    return hmac.digest(key, body, "sha256") + body


def decode(datagram: bytes, key: bytes, *, max_clock_skew: float, now: float | None = None) -> Packet:
    if not MAC_SIZE < len(datagram) <= MAX_DATAGRAM_SIZE:
        raise ProtocolError(f"bad datagram size {len(datagram)}")

    mac, body = datagram[:MAC_SIZE], datagram[MAC_SIZE:]
    # Verify the MAC *before* parsing anything (constant-time compare).
    if not hmac.compare_digest(mac, hmac.digest(key, body, "sha256")):
        raise AuthenticationError("bad MAC")

    try:
        obj = json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProtocolError("body is not valid JSON") from exc
    if not isinstance(obj, dict) or obj.get("v") != VERSION:
        raise ProtocolError("unsupported protocol version")

    try:
        kind = Kind(obj["k"])
        sid, seq, ts = obj["s"], obj["n"], obj["t"]
        payload = obj.get("p", "")
    except (KeyError, ValueError) as exc:
        raise ProtocolError("missing or invalid field") from exc

    if not (isinstance(sid, str) and 0 < len(sid) <= MAX_SID_LEN):
        raise ProtocolError("invalid session id")
    if type(seq) is not int or not 0 <= seq <= MAX_SEQ:
        raise ProtocolError("invalid sequence number")
    if type(ts) not in (int, float) or not math.isfinite(ts):
        raise ProtocolError("invalid timestamp")
    if not isinstance(payload, str):
        raise ProtocolError("payload must be a string")

    if max_clock_skew > 0:
        now = time.time() if now is None else now
        if abs(now - ts) > max_clock_skew:
            raise StalePacketError(f"timestamp off by {now - ts:+.1f}s")

    return Packet(kind=kind, sid=sid, seq=seq, ts=float(ts), payload=payload)
