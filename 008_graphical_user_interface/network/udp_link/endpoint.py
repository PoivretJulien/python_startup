"""Reliable, authenticated datagram endpoint shared by server and client.

Techniques used (the classic "reliable UDP" toolkit):
    * sequence numbers on every reliable packet (random start, uint32 wrap)
    * positive acknowledgment (ACK echoes the sequence number)
    * retransmission with exponential backoff + jitter, bounded retries
    * receiver-side de-duplication per (peer, session) -> at-most-once delivery
      to the application, while still re-ACKing duplicates (lost-ACK case)
    * flow control: bounded number of in-flight sends
    * HMAC authentication + replay window (see protocol.py)
    * peer table with TTL and a hard cap; unauthenticated packets never
      create state and are never answered (no amplification / reflection)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import secrets
import time
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from .config import LinkConfig
from .protocol import MAX_SEQ, Kind, Packet, ProtocolError, StalePacketError, decode, encode

log = logging.getLogger(__name__)

Addr = tuple[str, int]

_DEDUP_WINDOW = 1024


class DeliveryError(ConnectionError):
    """The peer did not acknowledge a packet after all retries."""


@dataclass(frozen=True, slots=True)
class Message:
    text: str
    peer: Addr
    seq: int
    sent_at: float  # sender's wall clock (epoch seconds)
    received_at: datetime


MessageHandler = Callable[[Message], Awaitable[None] | None]


@dataclass(slots=True)
class LinkStats:
    sent: int = 0
    delivered: int = 0  # DATA handed to the application
    retransmits: int = 0
    acked: int = 0
    duplicates: int = 0
    rejected: int = 0  # bad MAC / malformed / stale / not allowed
    delivery_failures: int = 0


class _SeenWindow:
    """Remembers the last N sequence numbers seen from one peer session."""

    __slots__ = ("_order", "_seen")

    def __init__(self) -> None:
        self._order: deque[int] = deque()
        self._seen: set[int] = set()

    def add(self, seq: int) -> bool:
        """Return True if `seq` is new."""
        if seq in self._seen:
            return False
        self._seen.add(seq)
        self._order.append(seq)
        if len(self._order) > _DEDUP_WINDOW:
            self._seen.discard(self._order.popleft())
        return True


class ReliableEndpoint(asyncio.DatagramProtocol):
    def __init__(
        self,
        config: LinkConfig,
        *,
        name: str,
        on_message: MessageHandler | None = None,
        peer_filter: Callable[[Addr], bool] | None = None,
    ) -> None:
        self.config = config
        self.name = name
        self.sid = secrets.token_hex(4)
        self.stats = LinkStats()
        self.received: list[Message] = []
        self.peers: dict[Addr, float] = {}  # addr -> last seen (monotonic)

        self._on_message = on_message
        self._peer_filter = peer_filter or (lambda addr: config.is_allowed(addr[0]))
        self._seq = secrets.randbelow(MAX_SEQ)
        self._transport: asyncio.DatagramTransport | None = None
        self._pending: dict[int, tuple[Addr, asyncio.Future[None]]] = {}
        self._dedup: OrderedDict[tuple[Addr, str], _SeenWindow] = OrderedDict()
        self._inflight = asyncio.Semaphore(config.max_in_flight)
        self._new_message = asyncio.Event()
        self._closed = asyncio.Event()
        self._handler_tasks: set[asyncio.Task[None]] = set()
        self._last_skew_warning = 0.0

    # ------------------------------------------------------------------ asyncio callbacks
    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        assert isinstance(transport, asyncio.DatagramTransport)
        self._transport = transport

    def connection_lost(self, exc: Exception | None) -> None:
        self._fail_pending(ConnectionError("endpoint closed"))
        self._closed.set()

    def error_received(self, exc: Exception) -> None:
        # e.g. ICMP port unreachable when the peer is down; retries handle it.
        log.debug("[%s] socket error: %s", self.name, exc)

    def datagram_received(self, data: bytes, addr: Addr) -> None:  # type: ignore[override]
        addr = (addr[0], addr[1])  # normalise (IPv6 tuples carry 4 items)
        if not self._peer_filter(addr):
            self.stats.rejected += 1
            return
        try:
            pkt = decode(data, self.config.key, max_clock_skew=self.config.max_clock_skew)
        except StalePacketError as exc:
            self.stats.rejected += 1
            self._warn_skew(addr, exc)
            return
        except ProtocolError as exc:
            self.stats.rejected += 1
            log.debug("[%s] dropped datagram from %s:%d: %s", self.name, *addr, exc)
            return

        if pkt.sid == self.sid:  # our own packet looped back
            return

        if pkt.kind is Kind.ACK:
            self._handle_ack(pkt, addr)
            return

        if not self._register_peer(addr):
            return  # peer table full: don't ACK, sender will give up

        self._send_raw(Packet(Kind.ACK, self.sid, pkt.seq, time.time()), addr)

        if not self._seen_window(addr, pkt.sid).add(pkt.seq):
            self.stats.duplicates += 1  # our previous ACK was lost; re-ACKed above
            return

        match pkt.kind:
            case Kind.DATA:
                self._deliver(pkt, addr)
            case Kind.BYE:
                self.peers.pop(addr, None)
                log.info("[%s] peer %s:%d left", self.name, *addr)
            case Kind.HELLO | Kind.PING:
                pass  # liveness only; _register_peer already refreshed it

    # ------------------------------------------------------------------ sending
    async def send_reliable(self, kind: Kind, text: str, addr: Addr) -> None:
        """Send and wait for the ACK. Raises DeliveryError after all retries."""
        if not isinstance(text, str):
            raise TypeError(f"payload must be str, not {type(text).__name__}")
        transport = self._require_transport()
        loop = asyncio.get_running_loop()
        cfg = self.config

        async with self._inflight:
            seq = self._next_seq()
            # Encode once up-front so an oversized payload fails immediately.
            encode(Packet(kind, self.sid, seq, time.time(), text), cfg.key)

            fut: asyncio.Future[None] = loop.create_future()
            self._pending[seq] = (addr, fut)
            try:
                timeout = cfg.ack_timeout
                for attempt in range(cfg.max_retries + 1):
                    if attempt:
                        self.stats.retransmits += 1
                    # Fresh timestamp each try keeps retries inside the replay window.
                    self._send_raw(Packet(kind, self.sid, seq, time.time(), text), addr, transport)
                    try:
                        async with asyncio.timeout(timeout):
                            await asyncio.shield(fut)
                        return
                    except TimeoutError:
                        timeout = min(timeout * 2, cfg.max_backoff) * random.uniform(0.8, 1.2)
                self.stats.delivery_failures += 1
                raise DeliveryError(
                    f"no ACK from {addr[0]}:{addr[1]} for seq {seq} after {cfg.max_retries + 1} attempts"
                )
            finally:
                self._pending.pop(seq, None)
                if not fut.done():
                    fut.cancel()

    def send_unreliable(self, kind: Kind, text: str, addr: Addr) -> None:
        """Fire-and-forget (used for BYE)."""
        self._send_raw(Packet(kind, self.sid, self._next_seq(), time.time(), text), addr)

    # ------------------------------------------------------------------ receiving
    async def recv(self) -> Message:
        """Pop the oldest stacked message, waiting if the stack is empty."""
        while not self.received:
            if self._closed.is_set():
                raise ConnectionError("endpoint closed")
            self._new_message.clear()
            closed = asyncio.ensure_future(self._closed.wait())
            new = asyncio.ensure_future(self._new_message.wait())
            try:
                await asyncio.wait({closed, new}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                closed.cancel()
                new.cancel()
        return self.received.pop(0)

    def drain(self) -> list[Message]:
        """Take every stacked message at once (oldest first) and clear the stack."""
        items, self.received = self.received, []
        return items

    # ------------------------------------------------------------------ peers
    def prune_peers(self) -> list[Addr]:
        if self.config.peer_ttl <= 0:  # idle mode: peers never time out
            return []
        now = time.monotonic()
        expired = [a for a, seen in self.peers.items() if now - seen > self.config.peer_ttl]
        for addr in expired:
            del self.peers[addr]
            log.info("[%s] peer %s:%d timed out", self.name, *addr)
        return expired

    def forget_peer(self, addr: Addr) -> None:
        self.peers.pop(addr, None)

    # ------------------------------------------------------------------ lifecycle
    @property
    def local_address(self) -> Addr:
        host, port, *_ = self._require_transport().get_extra_info("sockname")
        return host, port

    def close(self) -> None:
        self._fail_pending(ConnectionError("endpoint closed"))
        if self._transport is not None:
            self._transport.close()

    async def wait_closed(self) -> None:
        await self._closed.wait()
        if self._handler_tasks:
            await asyncio.gather(*self._handler_tasks, return_exceptions=True)

    # ------------------------------------------------------------------ internals
    def _require_transport(self) -> asyncio.DatagramTransport:
        if self._transport is None or self._transport.is_closing():
            raise ConnectionError("endpoint is not open")
        return self._transport

    def _next_seq(self) -> int:
        self._seq = (self._seq + 1) & MAX_SEQ
        return self._seq

    def _send_raw(self, pkt: Packet, addr: Addr, transport: asyncio.DatagramTransport | None = None) -> None:
        transport = transport or self._transport
        if transport is None or transport.is_closing():
            return
        transport.sendto(encode(pkt, self.config.key), addr)
        self.stats.sent += 1

    def _handle_ack(self, pkt: Packet, addr: Addr) -> None:
        entry = self._pending.get(pkt.seq)
        if entry is None:
            return  # late/duplicate ACK
        expected_addr, fut = entry
        if expected_addr != addr or fut.done():
            return
        self.stats.acked += 1
        if addr in self.peers:
            self.peers[addr] = time.monotonic()
        fut.set_result(None)

    def _register_peer(self, addr: Addr) -> bool:
        if addr in self.peers:
            self.peers[addr] = time.monotonic()
            return True
        self.prune_peers()
        if len(self.peers) >= self.config.max_peers:
            log.warning("[%s] peer table full (%d), ignoring %s:%d", self.name, self.config.max_peers, *addr)
            return False
        self.peers[addr] = time.monotonic()
        log.info("[%s] new peer %s:%d", self.name, *addr)
        return True

    def _seen_window(self, addr: Addr, sid: str) -> _SeenWindow:
        key = (addr, sid)
        window = self._dedup.get(key)
        if window is None:
            window = self._dedup[key] = _SeenWindow()
            while len(self._dedup) > self.config.max_peers * 4:
                self._dedup.popitem(last=False)
        else:
            self._dedup.move_to_end(key)
        return window

    def _deliver(self, pkt: Packet, addr: Addr) -> None:
        msg = Message(pkt.payload, addr, pkt.seq, pkt.ts, datetime.now(UTC))
        self.received.append(msg)
        if (overflow := len(self.received) - self.config.inbox_limit) > 0:
            del self.received[:overflow]
            log.warning("[%s] inbox full, dropped %d oldest message(s)", self.name, overflow)
        self.stats.delivered += 1
        self._new_message.set()

        if self._on_message is None:
            return
        try:
            result = self._on_message(msg)
            if inspect.isawaitable(result):
                task = asyncio.ensure_future(result)
                self._handler_tasks.add(task)
                task.add_done_callback(self._handler_done)
        except Exception:
            log.exception("[%s] on_message handler failed", self.name)

    def _handler_done(self, task: asyncio.Task[None]) -> None:
        self._handler_tasks.discard(task)
        if not task.cancelled() and (exc := task.exception()):
            log.error("[%s] on_message handler failed", self.name, exc_info=exc)

    def _fail_pending(self, exc: Exception) -> None:
        for _, fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
                fut.exception()  # mark retrieved: avoid "never retrieved" noise

    def _warn_skew(self, addr: Addr, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last_skew_warning > 10:
            self._last_skew_warning = now
            log.warning(
                "[%s] rejected packet from %s:%d (%s). Clocks out of sync? "
                "Enable NTP on both machines or set max_clock_skew = 0.",
                self.name, *addr, exc,
            )


class LinkBase:
    """Shared public API for UDPServer and UDPClient."""

    def __init__(self, config: LinkConfig, *, on_message: MessageHandler | None = None) -> None:
        self.config = config
        self._on_message = on_message
        self._ep: ReliableEndpoint | None = None

    def _endpoint(self) -> ReliableEndpoint:
        if self._ep is None:
            raise RuntimeError(f"{type(self).__name__} is not started")
        return self._ep

    @property
    def received(self) -> list[Message]:
        """The stack of received messages (oldest first). Safe to read/iterate."""
        return self._endpoint().received

    def drain(self) -> list[Message]:
        """Take all stacked messages and clear the stack."""
        return self._endpoint().drain()

    async def recv(self, timeout: float | None = None) -> Message:
        """Pop the oldest message, waiting up to `timeout` seconds."""
        async with asyncio.timeout(timeout):
            return await self._endpoint().recv()

    async def messages(self) -> AsyncIterator[Message]:
        """`async for msg in link.messages(): ...` until the link closes."""
        while True:
            try:
                yield await self._endpoint().recv()
            except ConnectionError:
                return

    @property
    def stats(self) -> LinkStats:
        return self._endpoint().stats

    @property
    def address(self) -> Addr:
        return self._endpoint().local_address
