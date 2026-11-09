"""Video client (runs on the PC): subscribes and reassembles JPEG frames.

    async with VideoClient(link_cfg, video_cfg) as client:
        async for frame in client.stream():       # always the newest frame
            jpeg_bytes = frame.data
        client.frames                              # the N most recent frames
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import time
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from types import TracebackType

from udp_link import LinkConfig

from .config import VideoConfig, make_udp_socket
from .protocol import (
    COOKIE_SIZE,
    MAX_CHUNK_PAYLOAD,
    Chunk,
    Kind,
    VideoProtocolError,
    decode,
    derive_video_key,
    encode,
)

log = logging.getLogger(__name__)

Addr = tuple[str, int]
SUBSCRIBE_RETRY = 2.0  # idle mode: seconds between SUBs until the server confirms


@dataclass(frozen=True, slots=True)
class Frame:
    frame_id: int
    data: bytes  # JPEG
    captured_at: float  # server wall clock
    received_at: float  # local wall clock

    @property
    def latency(self) -> float:
        """Capture-to-receive delay in seconds (needs NTP-synced clocks)."""
        return self.received_at - self.captured_at


@dataclass(slots=True)
class VideoClientStats:
    frames: int = 0
    bytes: int = 0
    incomplete: int = 0  # frames dropped because a chunk was lost
    late: int = 0  # chunks of frames older than the last delivered one
    rejected: int = 0


class _Partial:
    __slots__ = ("count", "have", "parts", "ts")

    def __init__(self, count: int, ts: float) -> None:
        self.count, self.ts, self.have = count, ts, 0
        self.parts: list[bytes | None] = [None] * count

    def add(self, index: int, payload: bytes) -> bool:
        if self.parts[index] is None:
            self.parts[index] = payload
            self.have += 1
        return self.have == self.count

    def join(self) -> bytes:
        return b"".join(p for p in self.parts if p is not None)


class _Protocol(asyncio.DatagramProtocol):
    def __init__(self, client: VideoClient) -> None:
        self.client = client

    def datagram_received(self, data: bytes, addr: Addr) -> None:  # type: ignore[override]
        self.client._on_datagram(data, (addr[0], addr[1]))

    def error_received(self, exc: Exception) -> None:
        log.debug("socket error: %s", exc)


class VideoClient:
    def __init__(
        self,
        link: LinkConfig,
        video: VideoConfig | None = None,
        *,
        on_frame: Callable[[Frame], None] | None = None,
    ) -> None:
        self.link = link
        self.video = video or VideoConfig()
        self.stats = VideoClientStats()
        self.frames: deque[Frame] = deque(maxlen=self.video.buffer_frames)
        self._on_frame = on_frame
        self._key = derive_video_key(link.key)
        self._cookie = bytes(COOKIE_SIZE)
        self._server: Addr | None = None
        self._transport: asyncio.DatagramTransport | None = None
        self._sub_task: asyncio.Task[None] | None = None
        self._session: bytes | None = None
        self._last_id = 0
        self._last_ts = 0.0
        self._partial: OrderedDict[int, _Partial] = OrderedDict()
        self._max_chunks = self.video.max_frame_bytes // MAX_CHUNK_PAYLOAD + 1
        self._delivered = 0
        self._consumed = 0
        self._new_frame = asyncio.Event()
        self._subscribed = asyncio.Event()  # set by the server's SUBACK (or by video arriving)
        self._last_sub = 0.0  # monotonic time of our last SUB

    # ------------------------------------------------------------------ lifecycle
    async def connect(self) -> None:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(
            self.link.server_ip, self.video.port, family=socket.AF_INET, type=socket.SOCK_DGRAM
        )
        self._server = (infos[0][4][0], infos[0][4][1])
        sock = make_udp_socket((self.link.client_bind_ip, 0), self.video.socket_buffer)
        self._transport, _ = await loop.create_datagram_endpoint(lambda: _Protocol(self), sock=sock)
        self._sub_task = asyncio.create_task(self._subscribe_loop(), name="video-subscribe")
        log.info("subscribing to video at %s:%d", *self._server)

    async def close(self) -> None:
        if self._sub_task:
            self._sub_task.cancel()
            try:
                await self._sub_task
            except asyncio.CancelledError:
                pass
        if self._transport is not None:
            self._send(Kind.BYE)
            self._transport.close()
            self._transport = None
            log.info("video client closed")

    async def __aenter__(self) -> VideoClient:
        await self.connect()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.close()

    # ------------------------------------------------------------------ consuming frames
    @property
    def latest(self) -> Frame | None:
        return self.frames[-1] if self.frames else None

    async def next_frame(self, timeout: float | None = None) -> Frame:
        """Wait for a frame newer than the last one returned; skip any backlog."""
        async with asyncio.timeout(timeout):
            while self._delivered == self._consumed:
                self._new_frame.clear()
                await self._new_frame.wait()
        self._consumed = self._delivered
        return self.frames[-1]

    @property
    def subscribed(self) -> bool:
        """True once the server has confirmed the subscription (SUBACK) or video arrived."""
        return self._subscribed.is_set()

    async def wait_for_frame(self, timeout: float | None = None) -> None:
        """Wait until a new frame arrives, without consuming it.

        Unlike next_frame(), the frame is still returned by the next
        next_frame()/stream() call, so this can run beside a viewer (run_client
        uses it to log statistics only while video is flowing).
        """
        seen = self._delivered
        async with asyncio.timeout(timeout):
            while self._delivered == seen:
                self._new_frame.clear()
                await self._new_frame.wait()

    async def stream(self) -> AsyncIterator[Frame]:
        """`async for frame in client.stream()`: newest frames, never a backlog."""
        while True:
            yield await self.next_frame()

    # ------------------------------------------------------------------ internals
    async def _subscribe_loop(self) -> None:
        if self.video.subscribe_interval > 0:
            # Normal mode: a SUB every subscribe_interval keeps the subscription alive.
            while True:
                self._send(Kind.SUB)
                await asyncio.sleep(self.video.subscribe_interval)
        # Idle mode: subscribe until the server confirms, then stop. From then
        # on a SUB is sent only while frames arrive (see _on_datagram), so
        # nothing runs while the server isn't streaming.
        while not self._subscribed.is_set():
            self._send(Kind.SUB)
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(SUBSCRIBE_RETRY):
                    await self._subscribed.wait()
        log.info("subscribed; keep-alive off (idle mode): the client wakes only for video")

    def _send(self, kind: Kind) -> None:
        if self._transport is None or self._transport.is_closing() or self._server is None:
            return
        chunk = Chunk(kind, bytes(4), ts=time.time(), payload=self._cookie)
        self._transport.sendto(encode(chunk, self._key), self._server)
        if kind is Kind.SUB:
            self._last_sub = time.monotonic()

    def _on_datagram(self, data: bytes, addr: Addr) -> None:
        if addr != self._server:
            self.stats.rejected += 1
            return
        try:
            chunk = decode(data, self._key, max_clock_skew=self.link.max_clock_skew)
        except VideoProtocolError as exc:
            self.stats.rejected += 1
            log.debug("dropped packet: %s", exc)
            return
        match chunk.kind:
            case Kind.CHALLENGE if len(chunk.payload) == COOKIE_SIZE:
                if chunk.payload != self._cookie:
                    self._cookie = chunk.payload
                    self._send(Kind.SUB)  # answer right away, don't wait a period
            case Kind.SUBACK:
                if not self._subscribed.is_set():
                    log.info("subscribed to video at %s:%d", *self._server)  # type: ignore[str-format]
                self._subscribed.set()
            case Kind.FRAME:
                self._subscribed.set()
                # Refresh the subscription while video flows, whatever the
                # keep-alive setting: this is what keeps an idle-mode client
                # subscribed. With the default 1 s keep-alive it never triggers.
                if time.monotonic() - self._last_sub > self.video.subscriber_ttl / 3:
                    self._send(Kind.SUB)
                self._on_chunk(chunk)

    def _on_chunk(self, chunk: Chunk) -> None:
        if chunk.session != self._session:
            # Accept a new server session (restart) only if it is newer: this
            # rejects replays of an old session's packets.
            if self._session is not None and chunk.ts <= self._last_ts:
                self.stats.rejected += 1
                return
            if self._session is not None:
                log.info("server restarted (new video session)")
            self._session, self._last_id = chunk.session, 0
            self.stats.incomplete += len(self._partial)
            self._partial.clear()

        if chunk.frame_id <= self._last_id:
            self.stats.late += 1
            return
        if chunk.count > self._max_chunks:
            self.stats.rejected += 1
            return

        partial = self._partial.get(chunk.frame_id)
        if partial is None:
            partial = self._partial[chunk.frame_id] = _Partial(chunk.count, chunk.ts)
            while len(self._partial) > self.video.max_partial_frames:
                self._partial.popitem(last=False)
                self.stats.incomplete += 1
        elif partial.count != chunk.count:
            self.stats.rejected += 1
            return

        if partial.add(chunk.index, chunk.payload):
            del self._partial[chunk.frame_id]
            for old in [fid for fid in self._partial if fid < chunk.frame_id]:
                del self._partial[old]  # older frames can no longer be shown
                self.stats.incomplete += 1
            self._last_id, self._last_ts = chunk.frame_id, chunk.ts
            self._deliver(Frame(chunk.frame_id, partial.join(), chunk.ts, time.time()))

    def _deliver(self, frame: Frame) -> None:
        if self.stats.frames == 0:
            log.info("receiving video from %s:%d", *self._server)  # type: ignore[str-format]
        self.frames.append(frame)
        self.stats.frames += 1
        self.stats.bytes += len(frame.data)
        self._delivered += 1
        self._new_frame.set()
        if self._on_frame is not None:
            try:
                self._on_frame(frame)
            except Exception:
                log.exception("on_frame handler failed")
