"""udp_link for programs that are not async: a window, a game loop, a script.

``UDPServer`` / ``UDPClient`` live inside an asyncio event loop. A program that
already owns its main thread (a GLFW window blocked in ``wait_events``, for
instance) cannot also run that loop there, so ``LinkThread`` runs it in a
background thread and leaves two plain calls::

    from udp_link.bridge import LinkThread

    def on_text(text: str) -> None:      # called for every received string
        ...                              # (from the link thread: keep it short)

    link = LinkThread("server", on_text=on_text)   # or "client"
    link.start()
    link.send("x=120;y=80")              # from any thread, never blocks
    ...
    link.close()

Both roles send and receive, so the link is bi-directional:

* ``"server"`` listens on ``bind_ip:port``; ``send`` goes to **every** connected
  client (``UDPServer.broadcast``).
* ``"client"`` connects to ``server_ip:port`` (it keeps retrying until the
  server is up); ``send`` goes to the server.

The configuration is the usual ``link.toml`` (see ``link.example.toml``); pass a
path, a ready ``LinkConfig``, or nothing to use the default lookup.

``latest_only=True`` (the default) is made for *state*: a cursor position, a
slider value. While one string is on its way and waiting for its ACK, newer
ones replace each other, so the receiver always ends on the last value and a
slow link never builds a backlog. Use ``latest_only=False`` for *events* that
must all arrive, in order.

Nothing here polls: the thread sleeps in the OS until a datagram arrives or
``send`` is called (plus the link's own heartbeat, see ``heartbeat_interval``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections import deque
from collections.abc import Callable

from .client import UDPClient
from .config import LinkConfig
from .endpoint import DeliveryError, Message
from .protocol import PayloadTooLargeError
from .server import UDPServer

log = logging.getLogger(__name__)

TextHandler = Callable[[str], None]


class LinkThread:
    """A ``UDPServer`` or ``UDPClient`` running in its own thread."""

    def __init__(
        self,
        role: str,
        config: LinkConfig | str | os.PathLike[str] | None = None,
        *,
        on_text: TextHandler | None = None,
        latest_only: bool = True,
    ) -> None:
        if role not in ("server", "client"):
            raise ValueError(f"role must be 'server' or 'client', not {role!r}")
        self.role = role
        self.config = config if isinstance(config, LinkConfig) else LinkConfig.load(config)
        self.on_text = on_text
        self.connected = False  # server: listening / client: handshake done
        self.error: BaseException | None = None  # why the link stopped, if it did

        self._outbox: deque[str] = deque(maxlen=1 if latest_only else None)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._wake: asyncio.Event | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"udp-link-{role}", daemon=True)

    # ------------------------------------------------------------ any thread
    def start(self) -> LinkThread:
        """Start the link thread and return at once (connecting goes on behind)."""
        self._thread.start()
        self._ready.wait()
        return self

    def send(self, text: str) -> None:
        """Queue ``text`` for sending. Never blocks; safe from any thread.

        Strings queued before the link is up are sent as soon as it is.
        """
        if not isinstance(text, str):
            raise TypeError(f"payload must be str, not {type(text).__name__}")
        self._outbox.append(text)
        self._notify()

    def close(self, timeout: float = 2.0) -> None:
        """Stop the link (the client says BYE) and wait for the thread to end."""
        loop, task = self._loop, self._task
        if loop is not None and task is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:  # the loop closed in the meantime
                pass
        if self._thread.is_alive():
            self._thread.join(timeout)

    def __enter__(self) -> LinkThread:
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.close()

    def _notify(self) -> None:
        loop, wake = self._loop, self._wake
        if loop is not None and wake is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(wake.set)
            except RuntimeError:
                pass

    # ----------------------------------------------------------- link thread
    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as exc:  # e.g. the port is already in use
            self.error = exc
            log.error("udp link (%s) stopped: %s", self.role, exc)
        finally:
            self.connected = False
            self._ready.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        self._wake = asyncio.Event()
        if self._outbox:
            self._wake.set()  # strings queued before start()
        self._ready.set()

        server = self.role == "server"
        link = (UDPServer if server else UDPClient)(self.config, on_message=self._deliver)
        try:
            async with link:
                self.connected = True
                while True:
                    await self._wake.wait()
                    self._wake.clear()
                    while self._outbox:
                        text = self._outbox.popleft()
                        try:
                            if server:
                                await link.broadcast(text)  # type: ignore[union-attr]
                            else:
                                await link.send(text)  # type: ignore[union-attr]
                        except (DeliveryError, PayloadTooLargeError) as exc:
                            log.warning("udp link (%s): %r not delivered: %s", self.role, text, exc)
        except asyncio.CancelledError:
            pass  # close(): the normal way out

    def _deliver(self, msg: Message) -> None:
        if self.on_text is not None:
            self.on_text(msg.text)
