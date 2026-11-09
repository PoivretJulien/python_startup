"""UDP client side (e.g. runs on your PC).

    async with UDPClient(LinkConfig.load()) as client:   # HELLO handshake
        await client.send("led=on")                       # reliable, ACKed
        async for msg in client.messages():               # strings from server
            print(msg.text)
        # or batch-process the stack: for msg in client.drain(): ...
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
from types import TracebackType

from .config import LinkConfig
from .endpoint import Addr, DeliveryError, LinkBase, MessageHandler, ReliableEndpoint
from .protocol import Kind

log = logging.getLogger(__name__)


class UDPClient(LinkBase):
    def __init__(
        self,
        config: LinkConfig,
        *,
        on_message: MessageHandler | None = None,
        wait_for_server: bool = True,
    ) -> None:
        super().__init__(config, on_message=on_message)
        self.wait_for_server = wait_for_server
        self.connected = False
        self._server: Addr | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None

    @property
    def server(self) -> Addr:
        if self._server is None:
            raise RuntimeError("client is not connected")
        return self._server

    async def connect(self) -> None:
        """Open the socket and perform the HELLO handshake.

        With wait_for_server=True (default) this retries until the server
        answers, so the client can be started before server_side is up.
        """
        if self._ep is not None:
            raise RuntimeError("client already connected")
        loop = asyncio.get_running_loop()
        cfg = self.config

        infos = await loop.getaddrinfo(cfg.server_ip, cfg.port, family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self._server = server = infos[0][4][0], infos[0][4][1]

        _, ep = await loop.create_datagram_endpoint(
            lambda: ReliableEndpoint(
                cfg,
                name="client",
                on_message=self._on_message,
                peer_filter=lambda addr: addr == server,  # only talk to our server
            ),
            local_addr=(cfg.client_bind_ip, 0),
            family=socket.AF_INET,
        )
        self._ep = ep
        try:
            await self._handshake()
        except BaseException:
            await self.close()
            raise
        self._heartbeat_task = asyncio.create_task(self._heartbeat(), name="udp-client-heartbeat")

    async def _handshake(self) -> None:
        ep, server = self._endpoint(), self.server
        while True:
            try:
                await ep.send_reliable(Kind.HELLO, "", server)
            except DeliveryError:
                if not self.wait_for_server:
                    raise
                log.warning(
                    "server %s:%d not answering (down, wrong IP/port or wrong secret); retrying...", *server
                )
                await asyncio.sleep(self.config.heartbeat_interval)
            else:
                self.connected = True
                log.info("connected to %s:%d from %s:%d (session %s)", *server, *ep.local_address, ep.sid)
                return

    async def _heartbeat(self) -> None:
        """Keep the server's peer entry alive and detect link loss / recovery."""
        ep, server = self._endpoint(), self.server
        while True:
            await asyncio.sleep(self.config.heartbeat_interval)
            try:
                await ep.send_reliable(Kind.PING, "", server)
            except DeliveryError:
                if self.connected:
                    log.warning("lost contact with server %s:%d", *server)
                self.connected = False
            except ConnectionError:
                return
            else:
                if not self.connected:
                    log.info("link to server %s:%d restored", *server)
                self.connected = True

    async def send(self, text: str) -> None:
        """Reliable send to the server. Raises DeliveryError if not ACKed."""
        await self._endpoint().send_reliable(Kind.DATA, text, self.server)

    async def close(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
        if self._ep is not None:
            with contextlib.suppress(Exception):
                self._ep.send_unreliable(Kind.BYE, "", self.server)
            self._ep.close()
            await self._ep.wait_closed()
            self.connected = False
            log.info("client closed")

    async def __aenter__(self) -> UDPClient:
        await self.connect()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.close()
