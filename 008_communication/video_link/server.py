"""Video server (runs on server_side): pushes JPEG frames to subscribed clients.

    async with VideoServer(link_cfg, video_cfg) as server:
        await server.publish(jpeg_bytes)          # sent to every subscriber
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import secrets
import time
from dataclasses import dataclass
from types import TracebackType

from udp_link import LinkConfig

from .config import VideoConfig, make_udp_socket
from .protocol import (
    COOKIE_SIZE,
    SESSION_SIZE,
    Chunk,
    Kind,
    VideoProtocolError,
    decode,
    derive_video_key,
    encode,
    split_frame,
)

log = logging.getLogger(__name__)

Addr = tuple[str, int]
_COOKIE_PERIOD = 30.0  # seconds; cookies stay valid for 30-60 s


@dataclass(slots=True)
class VideoServerStats:
    frames_sent: int = 0
    frames_skipped: int = 0  # dropped because the socket backlog was too big
    bytes_sent: int = 0
    rejected: int = 0


class _Protocol(asyncio.DatagramProtocol):
    def __init__(self, server: VideoServer) -> None:
        self.server = server

    def datagram_received(self, data: bytes, addr: Addr) -> None:  # type: ignore[override]
        self.server._on_datagram(data, (addr[0], addr[1]))

    def error_received(self, exc: Exception) -> None:
        log.debug("socket error: %s", exc)


class VideoServer:
    def __init__(self, link: LinkConfig, video: VideoConfig | None = None) -> None:
        self.link = link
        self.video = video or VideoConfig()
        self.stats = VideoServerStats()
        self.subscribers: dict[Addr, float] = {}  # addr -> last SUB (monotonic)
        self.session = secrets.token_bytes(SESSION_SIZE)
        self._key = derive_video_key(link.key)
        self._cookie_secret = secrets.token_bytes(32)
        self._frame_id = 0
        self._transport: asyncio.DatagramTransport | None = None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> Addr:
        sock = make_udp_socket((self.link.bind_ip, self.video.port), self.video.socket_buffer)
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(lambda: _Protocol(self), sock=sock)
        self._transport = transport
        log.info("video server listening on %s:%d", *self.address)
        return self.address

    async def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None
            log.info("video server closed")

    async def __aenter__(self) -> VideoServer:
        await self.start()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.close()

    @property
    def address(self) -> Addr:
        host, port, *_ = self._require().get_extra_info("sockname")
        return host, port

    # ------------------------------------------------------------------ publishing
    async def publish(self, jpeg: bytes, captured_at: float | None = None) -> int:
        """Send one encoded frame to every subscriber. Returns how many got it.

        Never blocks on the network: if the OS cannot keep up, the frame is
        skipped (latest-frame-wins) instead of building up latency.
        """
        transport = self._require()
        if len(jpeg) > self.video.max_frame_bytes:
            raise ValueError(f"frame is {len(jpeg)} bytes, max_frame_bytes is {self.video.max_frame_bytes}")
        self._prune()
        if not self.subscribers:
            return 0
        if transport.get_write_buffer_size() > self.video.max_write_buffer:
            self.stats.frames_skipped += 1
            return 0

        self._frame_id += 1
        ts = time.time() if captured_at is None else captured_at
        packets = [encode(c, self._key) for c in split_frame(jpeg, session=self.session, frame_id=self._frame_id, ts=ts)]
        targets = list(self.subscribers)
        for addr in targets:
            for packet in packets:
                transport.sendto(packet, addr)
        self.stats.frames_sent += 1
        self.stats.bytes_sent += sum(map(len, packets)) * len(targets)
        await asyncio.sleep(0)  # let the loop flush and handle incoming SUBs
        return len(targets)

    # ------------------------------------------------------------------ subscriptions
    def _on_datagram(self, data: bytes, addr: Addr) -> None:
        if not self.link.is_allowed(addr[0]):
            self.stats.rejected += 1
            return
        try:
            chunk = decode(data, self._key, max_clock_skew=self.link.max_clock_skew)
        except VideoProtocolError as exc:
            self.stats.rejected += 1
            log.debug("dropped packet from %s:%d: %s", *addr, exc)
            return
        if chunk.kind not in (Kind.SUB, Kind.BYE) or len(chunk.payload) != COOKIE_SIZE:
            return

        if not self._cookie_valid(addr, chunk.payload):
            if chunk.kind is Kind.SUB:
                # Reply is the same size as the request: no amplification.
                reply = Chunk(Kind.CHALLENGE, self.session, ts=time.time(), payload=self._cookie(addr, 0))
                self._require().sendto(encode(reply, self._key), addr)
            return

        if chunk.kind is Kind.BYE:
            if self.subscribers.pop(addr, None) is not None:
                log.info("subscriber %s:%d left", *addr)
            return

        if addr not in self.subscribers:
            self._prune()
            if len(self.subscribers) >= self.video.max_subscribers:
                log.warning("max_subscribers (%d) reached, ignoring %s:%d", self.video.max_subscribers, *addr)
                return
            log.info("new subscriber %s:%d", *addr)
        self.subscribers[addr] = time.monotonic()

    def _cookie(self, addr: Addr, age: int) -> bytes:
        bucket = int(time.monotonic() // _COOKIE_PERIOD) - age
        msg = f"{addr[0]}|{addr[1]}|{bucket}".encode()
        return hmac.digest(self._cookie_secret, msg, "sha256")[:COOKIE_SIZE]

    def _cookie_valid(self, addr: Addr, cookie: bytes) -> bool:
        return any(hmac.compare_digest(cookie, self._cookie(addr, age)) for age in (0, 1))

    def _prune(self) -> None:
        now = time.monotonic()
        for addr in [a for a, seen in self.subscribers.items() if now - seen > self.video.subscriber_ttl]:
            del self.subscribers[addr]
            log.info("subscriber %s:%d timed out", *addr)

    def _require(self) -> asyncio.DatagramTransport:
        if self._transport is None or self._transport.is_closing():
            raise ConnectionError("video server is not running")
        return self._transport
