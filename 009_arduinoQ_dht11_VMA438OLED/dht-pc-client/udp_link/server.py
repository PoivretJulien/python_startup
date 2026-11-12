"""UDP server side (runs on server_side).

    async with UDPServer(LinkConfig.load()) as server:
        await server.broadcast("temp=21.4;hum=40.2")   # push to every client
        msg = await server.recv()                     # strings sent by clients
        print(server.received)                        # ...or read the stack
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from types import TracebackType

from ._net import open_udp_socket
from .config import LinkConfig
from .endpoint import Addr, LinkBase, MessageHandler, ReliableEndpoint
from .protocol import Kind

log = logging.getLogger(__name__)


class UDPServer(LinkBase):
    def __init__(self, config: LinkConfig, *, on_message: MessageHandler | None = None) -> None:
        super().__init__(config, on_message=on_message)
        self._housekeeping_task: asyncio.Task[None] | None = None

    async def start(self) -> Addr:
        if self._ep is not None:
            raise RuntimeError("server already started")
        loop = asyncio.get_running_loop()
        cfg = self.config
        _, ep = await loop.create_datagram_endpoint(
            lambda: ReliableEndpoint(cfg, name="server", on_message=self._on_message),
            sock=open_udp_socket(cfg.bind_ip, cfg.port),
        )
        self._ep = ep
        if cfg.peer_ttl > 0:
            self._housekeeping_task = asyncio.create_task(self._housekeeping(), name="udp-server-housekeeping")
        else:
            # Idle mode: no timer. Clients are forgotten only on BYE or a failed send.
            log.info("peer expiry off (idle mode): the server wakes only for traffic")
        log.info("server listening on %s:%d (session %s)", *ep.local_address, ep.sid)
        return ep.local_address

    async def close(self) -> None:
        if self._housekeeping_task:
            self._housekeeping_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._housekeeping_task
        if self._ep is not None:
            self._ep.close()
            await self._ep.wait_closed()
            log.info("server closed")

    async def __aenter__(self) -> UDPServer:
        await self.start()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.close()

    @property
    def peers(self) -> list[Addr]:
        """Clients currently known (sent HELLO/PING/DATA within peer_ttl, or ever if peer_ttl is 0)."""
        return list(self._endpoint().peers)

    async def send(self, text: str, peer: Addr) -> None:
        """Reliable send to one client. Raises DeliveryError if not ACKed."""
        await self._endpoint().send_reliable(Kind.DATA, text, peer)

    async def broadcast(self, text: str) -> dict[Addr, bool]:
        """Reliable send to every known client concurrently -> {peer: delivered}.

        Clients that fail to ACK are dropped from the peer table; they are
        re-added automatically on their next heartbeat.
        """
        ep = self._endpoint()
        peers = list(ep.peers)
        results = await asyncio.gather(
            *(ep.send_reliable(Kind.DATA, text, p) for p in peers), return_exceptions=True
        )
        outcome: dict[Addr, bool] = {}
        for peer, res in zip(peers, results, strict=True):
            if isinstance(res, BaseException) and not isinstance(res, ConnectionError):
                raise res  # programming error (e.g. payload too large / not str)
            outcome[peer] = res is None
            if res is not None:
                log.warning("client %s:%d did not ACK: %s", *peer, res)
                ep.forget_peer(peer)
        return outcome

    async def _housekeeping(self) -> None:
        while True:
            await asyncio.sleep(self.config.peer_ttl / 2)
            self._endpoint().prune_peers()
