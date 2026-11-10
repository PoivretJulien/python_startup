"""Wire format for video_link: JPEG frames split into UDP chunks.

Every datagram is::

    +---------------------------+-------------------+------------------+
    | header (24 B, big-endian) | payload           | HMAC tag (16 B)  |
    +---------------------------+-------------------+------------------+

    header = magic "VL" | version u8 | kind u8 | session 4B |
             frame_id u32 | chunk_index u16 | chunk_count u16 | timestamp f64

Design (real-time media rules, unlike udp_link):
    * No ACK and no retransmission: a late frame is worthless. A frame with a
      missing chunk is dropped; the next frame replaces it.
    * Latest-frame-wins: frame ids only move forward; older or duplicate
      frames are discarded.
    * Every chunk is authenticated (truncated HMAC-SHA256, 128-bit tag). The
      video key is derived from the udp_link secret, so one secret serves both.
    * Address validation with a stateless cookie (like QUIC/DTLS): the server
      only streams to an address that proved it can receive packets there.
      This stops someone from replaying a SUB with a spoofed source address
      to flood another host with video.
    * Every accepted SUB is confirmed with a SUBACK of the same size (no
      amplification), so a client knows it is subscribed without waiting for
      video. That is what lets the keep-alive be switched off (idle mode).
"""

from __future__ import annotations

import hmac
import math
import struct
import time
from dataclasses import dataclass
from enum import IntEnum

MAGIC = b"VL"
VERSION = 1
HEADER = struct.Struct("!2sBB4sIHHd")
TAG_SIZE = 16
COOKIE_SIZE = 16
SESSION_SIZE = 4
MAX_DATAGRAM_SIZE = 1200  # below the IPv6 minimum MTU: no IP fragmentation
MAX_CHUNK_PAYLOAD = MAX_DATAGRAM_SIZE - HEADER.size - TAG_SIZE  # 1160 bytes
MAX_CHUNKS = 0xFFFF


class Kind(IntEnum):
    SUB = 1  # client -> server: subscribe / keep-alive (payload: cookie)
    CHALLENGE = 2  # server -> client: here is your address cookie
    FRAME = 3  # server -> client: one chunk of a JPEG frame
    BYE = 4  # client -> server: unsubscribe (payload: cookie)
    SUBACK = 5  # server -> client: subscription accepted (payload: the cookie, echoed)


class VideoProtocolError(ValueError):
    """Datagram is malformed, unauthenticated or stale."""


@dataclass(frozen=True, slots=True)
class Chunk:
    kind: Kind
    session: bytes
    frame_id: int = 0
    index: int = 0
    count: int = 1
    ts: float = 0.0
    payload: bytes = b""


def derive_video_key(link_key: bytes) -> bytes:
    """Separate key for video, derived from the udp_link key (key separation)."""
    return hmac.digest(link_key, b"video_link/v1", "sha256")


def encode(chunk: Chunk, key: bytes) -> bytes:
    body = (
        HEADER.pack(
            MAGIC, VERSION, chunk.kind, chunk.session, chunk.frame_id, chunk.index, chunk.count, chunk.ts
        )
        + chunk.payload
    )
    if len(body) + TAG_SIZE > MAX_DATAGRAM_SIZE:
        raise VideoProtocolError("chunk payload too large")
    return body + hmac.digest(key, body, "sha256")[:TAG_SIZE]


def decode(datagram: bytes, key: bytes, *, max_clock_skew: float, now: float | None = None) -> Chunk:
    if not HEADER.size + TAG_SIZE <= len(datagram) <= MAX_DATAGRAM_SIZE:
        raise VideoProtocolError(f"bad datagram size {len(datagram)}")
    body, tag = datagram[:-TAG_SIZE], datagram[-TAG_SIZE:]
    if not hmac.compare_digest(tag, hmac.digest(key, body, "sha256")[:TAG_SIZE]):
        raise VideoProtocolError("bad MAC")

    magic, version, kind, session, frame_id, index, count, ts = HEADER.unpack_from(body)
    if magic != MAGIC or version != VERSION:
        raise VideoProtocolError("not a video_link v1 packet")
    try:
        kind = Kind(kind)
    except ValueError as exc:
        raise VideoProtocolError(f"unknown kind {kind}") from exc
    if count < 1 or index >= count:
        raise VideoProtocolError("bad chunk index/count")
    if not math.isfinite(ts):
        raise VideoProtocolError("bad timestamp")
    if max_clock_skew > 0:
        now = time.time() if now is None else now
        if abs(now - ts) > max_clock_skew:
            raise VideoProtocolError(f"stale packet (timestamp off by {now - ts:+.1f}s)")
    return Chunk(kind, session, frame_id, index, count, ts, body[HEADER.size :])


def split_frame(data: bytes, *, session: bytes, frame_id: int, ts: float) -> list[Chunk]:
    count = max(1, math.ceil(len(data) / MAX_CHUNK_PAYLOAD))
    if count > MAX_CHUNKS:
        raise VideoProtocolError("frame too large")
    return [
        Chunk(Kind.FRAME, session, frame_id, i, count, ts, data[i * MAX_CHUNK_PAYLOAD : (i + 1) * MAX_CHUNK_PAYLOAD])
        for i in range(count)
    ]
